"""
analysis/analysis_a.py - Geliştirilmiş Ana Analiz Aggregator
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

from config import BotConfig, get_config
from utils.binance.binance_a import BinanceAPI

# Analiz modüllerini import et
from .causality import CausalityAnalyzer
from .derivs import compute_derivatives_sentiment
from .onchain import get_onchain_analyzer
from .orderflow import OrderflowAnalyzer
from .regime import get_regime_analyzer
from .risk import RiskManager
from .tremo import TremoAnalyzer
from .score import get_score_aggregator, ScoreConfig  # Yeni skor modülü

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Geliştirilmiş analiz sonuçları"""
    symbol: str
    timestamp: float
    module_scores: Dict[str, float]
    alpha_signal_score: float
    position_risk_score: float
    gnosis_signal: float
    recommendation: str
    position_size: float
    confidence: float
    market_regime: str
    timeframes: Dict[str, float]  # Çoklu timeframe analizi

class AnalysisAggregator:
    """Geliştirilmiş ana analiz aggregator"""
    
    _instance = None
    
    def __new__(cls, binance_api: BinanceAPI):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(binance_api)
        return cls._instance
    
    def _initialize(self, binance_api: BinanceAPI):
        self.binance = binance_api
        self.config = None
        self._cache = {}
        
        # Analiz modüllerini initialize et
        self.causality = CausalityAnalyzer()
        self.causality.set_binance_api(binance_api)
        
        self.onchain = get_onchain_analyzer(binance_api)
        self.orderflow = OrderflowAnalyzer(binance_api)
        self.regime = get_regime_analyzer(binance_api)
        self.risk = RiskManager(binance_api)
        self.tremo = TremoAnalyzer(binance_api)
        
        # Skor agregator
        self.score_aggregator = get_score_aggregator()
    
    async def _get_config(self):
        if self.config is None:
            self.config = await get_config()
        return self.config
    
    @lru_cache(maxsize=100)
    async def _get_cached_analysis(self, symbol: str, cache_key: str):
        """Geliştirilmiş cache mekanizması"""
        current_time = time.time()
        if cache_key in self._cache:
            cached_data, timestamp, symbol_cached = self._cache[cache_key]
            # Sembol değişmişse cache'i geçersiz kıl
            if symbol == symbol_cached and current_time - timestamp < 60:
                return cached_data
        return None
    
    async def run_multi_timeframe_analysis(self, symbol: str) -> Dict[str, float]:
        """Çoklu timeframe analizi"""
        timeframes = ["15m", "1h", "4h", "1d"]
        results = {}
        
        for tf in timeframes:
            try:
                # Regime analyzer multi-timeframe desteği
                if hasattr(self.regime, 'analyze'):
                    result = await self.regime.analyze(symbol, tf)
                    results[tf] = result.score
                else:
                    results[tf] = 0.0
            except Exception as e:
                logger.warning(f"Timeframe analiz hatası {symbol} {tf}: {e}")
                results[tf] = 0.0
                
        return results
    
    async def run_analysis(self, symbol: str) -> AnalysisResult:
        """Geliştirilmiş analiz metodu"""
        config = await self._get_config()
        cache_key = f"analysis_{symbol}_{int(time.time() // 60)}"  # Dakikalık cache
        
        # Cache kontrolü
        cached = await self._get_cached_analysis(symbol, cache_key)
        if cached:
            return cached
        
        module_scores = {}
        module_errors = {}
        
        try:
            # Tüm modülleri paralel çalıştır with timeout
            tasks = {
                "causality": asyncio.create_task(self._run_causality(symbol)),
                "derivs": asyncio.create_task(self._run_derivs(symbol)),
                "onchain": asyncio.create_task(self._run_onchain()),
                "orderflow": asyncio.create_task(self._run_orderflow(symbol)),
                "regime": asyncio.create_task(self._run_regime(symbol)),
                "tremo": asyncio.create_task(self._run_tremo(symbol)),
                "risk": asyncio.create_task(self._run_risk(symbol))
            }
            
            # Timeout ile çalıştır
            for name, task in tasks.items():
                try:
                    result = await asyncio.wait_for(task, timeout=30.0)
                    module_scores[name] = result
                except asyncio.TimeoutError:
                    logger.warning(f"{name} modülü timeout oldu")
                    module_scores[name] = 0.0
                    module_errors[name] = "timeout"
                except Exception as e:
                    logger.error(f"{name} modülü hatası: {e}")
                    module_scores[name] = 0.0
                    module_errors[name] = str(e)
        
        except Exception as e:
            logger.error(f"Analiz sırasında beklenmeyen hata: {e}")
            # Fallback scores
            module_scores = {name: 0.0 for name in tasks.keys()}
        
        # Skor agregasyonu
        score_result = self.score_aggregator.calculate_final_score(module_scores)
        
        # Risk skoru
        risk_result = await self.risk.combined_risk_score(symbol)
        position_risk_score = risk_result.get("score", 0.6)
        
        # Gnosis signal
        gnosis_signal = score_result["final_score"] * position_risk_score
        
        # Çoklu timeframe analizi
        timeframe_scores = await self.run_multi_timeframe_analysis(symbol)
        
        # Öneri ve pozisyon büyüklüğü
        recommendation, position_size = self._get_recommendation(
            gnosis_signal, 
            score_result["confidence"],
            config
        )
        
        # Piyasa rejimi
        market_regime = await self._get_market_regime(symbol)
        
        result = AnalysisResult(
            symbol=symbol,
            timestamp=time.time(),
            module_scores=module_scores,
            alpha_signal_score=score_result["raw_score"],
            position_risk_score=position_risk_score,
            gnosis_signal=gnosis_signal,
            recommendation=recommendation,
            position_size=position_size,
            confidence=score_result["confidence"],
            market_regime=market_regime,
            timeframes=timeframe_scores
        )
        
        # Cache'e kaydet
        self._cache[cache_key] = (result, time.time(), symbol)
        
        logger.info(f"✅ Analiz tamamlandı: {symbol} - Skor: {gnosis_signal:.3f} - Güven: {score_result['confidence']:.2f}")
        return result
    
    def _get_recommendation(self, gnosis_signal: float, confidence: float, config) -> Tuple[str, float]:
        """Geliştirilmiş öneri sistemi"""
        thresholds = config.SIGNAL_THRESHOLDS
        
        # Güven faktörü ile threshold ayarı
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5-1.0 arası
        adjusted_thresholds = {
            k: v * confidence_multiplier for k, v in thresholds.items()
        }
        
        if gnosis_signal >= adjusted_thresholds["strong_bull"]:
            return "STRONG_BUY", min(1.0, 0.8 + (confidence * 0.2))
        elif gnosis_signal >= adjusted_thresholds["bull"]:
            return "BUY", min(0.8, 0.5 + (confidence * 0.3))
        elif gnosis_signal <= adjusted_thresholds["strong_bear"]:
            return "STRONG_SELL", min(1.0, 0.8 + (confidence * 0.2))
        elif gnosis_signal <= adjusted_thresholds["bear"]:
            return "SELL", min(0.8, 0.5 + (confidence * 0.3))
        else:
            return "HOLD", 0.0
    
    async def _get_market_regime(self, symbol: str) -> str:
        """Piyasa rejimini belirle"""
        try:
            regime_result = await self.regime.analyze(symbol)
            score = getattr(regime_result, 'score', 0.0)
            
            if score >= 0.3:
                return "TREND_BULL"
            elif score <= -0.3:
                return "TREND_BEAR"
            else:
                return "RANGE"
        except Exception as e:
            logger.warning(f"Piyasa rejimi belirleme hatası: {e}")
            return "UNKNOWN"
    
    # Modül çalıştırma metodları (mevcut implementasyon korunacak)
   
    async def _run_causality(self, symbol: str) -> float:
        """Causality analizini çalıştır"""
        try:
            result = await self.causality.get_causality_score(symbol)
            return result.get("score", 0.0)
        except Exception as e:
            logger.error(f"Causality analiz hatası: {e}")
            return 0.0
    
    async def _run_derivs(self, symbol: str) -> float:
        """Derivatives analizini çalıştır"""
        try:
            result = await compute_derivatives_sentiment(self.binance, symbol)
            return result.get("combined_score", 0.0)
        except Exception as e:
            logger.error(f"Derivs analiz hatası: {e}")
            return 0.0
    
    async def _run_onchain(self) -> float:
        """On-chain analizini çalıştır"""
        try:
            result = await self.onchain.aggregate_score()
            return result.get("aggregate", 0.0)
        except Exception as e:
            logger.error(f"On-chain analiz hatası: {e}")
            return 0.0
    
    async def _run_orderflow(self, symbol: str) -> float:
        """Orderflow analizini çalıştır"""
        try:
            result = await self.orderflow.compute_orderflow_score(symbol)
            return result.get("pressure_score", 0.0)
        except Exception as e:
            logger.error(f"Orderflow analiz hatası: {e}")
            return 0.0
    
    async def _run_regime(self, symbol: str) -> float:
        """Regime analizini çalıştır"""
        try:
            #result = await self.regime.analyze(symbol)
            #return result.get("Score", 0.0)
            result = await self.regime.analyze(symbol)
            return result.score
        except Exception as e:
            logger.error(f"Regime analiz hatası: {e}")
            return 0.0
    
    async def _run_tremo(self, symbol: str) -> float:
        """Tremo analizini çalıştır"""
        try:
            result = await self.tremo.analyze(symbol)
            return result.signal_score
        except Exception as e:
            logger.error(f"Tremo analiz hatası: {e}")
            return 0.0


# Singleton instance
def get_analysis_aggregator(binance_api: BinanceAPI) -> AnalysisAggregator:
    return AnalysisAggregator(binance_api)