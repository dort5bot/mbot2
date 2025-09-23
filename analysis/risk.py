# analysis/risk.py
"""
analysis/risk.py
Glassnode API Key alın ve environment variable olarak ayarlayın
Risk parametrelerini bot stratejinize göre tuning edin
Cache timeout değerlerini ihtiyaca göre ayarlayın
Makro ağırlığını (macro_weight) backtest ile optimize edin
sağlam risk yönetimini koruyor
# ETF flow placeholder - gerçek API bulunana kadar
"""


from __future__ import annotations

import asyncio
import logging
import os
from math import erf, sqrt
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import pandas as pd

from .binance_a import BinanceAPI

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Config yönetimi için constants
DEFAULT_ATR_PERIOD = 14
DEFAULT_K_ATR_STOP = 3.0
DEFAULT_VAR_CONFIDENCE = 0.95
DEFAULT_MACRO_WEIGHT = 0.15
DEFAULT_MACRO_CACHE_TIMEOUT = 3600  # 1 saat cache

@dataclass
class RiskManagerConfig:
    """RiskManager yapılandırma sınıfı"""
    atr_period: int = DEFAULT_ATR_PERIOD
    k_atr_stop: float = DEFAULT_K_ATR_STOP
    var_confidence: float = DEFAULT_VAR_CONFIDENCE
    macro_weight: float = DEFAULT_MACRO_WEIGHT
    macro_cache_timeout: int = DEFAULT_MACRO_CACHE_TIMEOUT
    glassnode_api_key: Optional[str] = None

@dataclass
class MacroMarketSignal:
    """Makro piyasa sinyallerini tutan veri sınıfı"""
    ssr_score: float = 0.0  # -1 (bearish) to +1 (bullish)
    netflow_score: float = 0.0
    etf_flow_score: float = 0.0
    fear_greed_score: float = 0.0  # Yeni eklenen metrik
    overall_score: float = 0.0
    confidence: float = 0.0  # Sinyal güvenilirliği 0-1 arası
    timestamp: datetime = datetime.now()

class RiskManager:
    """
    Geliştirilmiş Risk Manager with macro market analysis integration.
    
    Yeni Özellikler:
    - Glassnode API entegrasyonu (SSR, Netflow için gerçek veri)
    - Fear & Greed Index entegrasyonu
    - Makro sinyallerin risk skoruna ağırlıklı entegrasyonu
    - Daha detaylı logging ve monitoring
    - Configurable parameters via dataclass
    """

    _instance: Optional[RiskManager] = None
    _initialized: bool = False

    def __new__(cls, binance: BinanceAPI, config: Optional[RiskManagerConfig] = None) -> RiskManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, binance: BinanceAPI, config: Optional[RiskManagerConfig] = None) -> None:
        if not self._initialized:
            self._initialize(binance, config)
            self._initialized = True

    def _initialize(self, binance: BinanceAPI, config: Optional[RiskManagerConfig]) -> None:
        self.binance = binance
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Config yönetimi
        self.config = config or RiskManagerConfig()
        
        # Environment variable'dan API key al
        if not self.config.glassnode_api_key:
            self.config.glassnode_api_key = os.getenv("GLASSNODE_API_KEY")
        
        # Cache mekanizmaları
        self._klines_cache: Dict[Tuple[str, str, int, bool], List[dict]] = {}
        self._macro_cache: Dict[str, Tuple[MacroMarketSignal, datetime]] = {}
        
        logger.debug("Geliştirilmiş RiskManager initialized")

    async def _ensure_session(self) -> None:
        """Session'ın hazır olduğundan emin ol"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    # -------------------------
    # GLASSNODE ENTEGRASYONU - gerçek implementasyonu
    # -------------------------
    async def _fetch_glassnode_data(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Glassnode API'den veri çekme"""
        if not self.config.glassnode_api_key:
            logger.warning("Glassnode API key not configured")
            return None

        try:
            await self._ensure_session()
            url = f"https://api.glassnode.com/v1/{endpoint}"
            params['api_key'] = self.config.glassnode_api_key
            
            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.warning(f"Glassnode API error: {response.status}")
                    return None
        except asyncio.TimeoutError:
            logger.warning("Glassnode API request timed out")
            return None
        except Exception as e:
            logger.error(f"Glassnode fetch error: {e}")
            return None

    async def get_ssr_metric(self) -> float:
        """Gerçek SSR metriği - Glassnode implementasyonu"""
        try:
            data = await self._fetch_glassnode_data("metrics/indicators/ssr", {
                'a': 'BTC',
                'i': '24h'
            })
            
            if data and len(data) > 0:
                latest_ssr = data[-1]['v']
                # Normalize: Tarihsel verilere göre ayarlanabilir
                if latest_ssr > 20: 
                    return -1.0
                elif latest_ssr < 5: 
                    return 1.0
                return (10 - latest_ssr) / 5
            return 0.0
        except Exception as e:
            logger.error(f"SSR metric error: {e}")
            return 0.0

    async def get_netflow_metric(self) -> float:
        """Gerçek Netflow metriği - Glassnode implementasyonu"""
        try:
            data = await self._fetch_glassnode_data("metrics/transactions/transfers_volume_exchanges_net", {
                'a': 'BTC',
                'i': '24h'
            })
            
            if data and len(data) > 0:
                latest_netflow = data[-1]['v']
                # Normalize: Tarihsel standart sapmaya göre ayarlanabilir
                if latest_netflow > 1000: 
                    return -1.0  # Büyük giriş → bearish
                elif latest_netflow < -1000: 
                    return 1.0  # Büyük çıkış → bullish
                return -latest_netflow / 1000  # Linear normalization
            return 0.0
        except Exception as e:
            logger.error(f"Netflow metric error: {e}")
            return 0.0

    async def get_fear_greed_index(self) -> float:
        """Fear & Greed Index - Alternative.me API'si"""
        try:
            await self._ensure_session()
            url = "https://api.alternative.me/fng/"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    data = await response.json()
                    score = int(data['data'][0]['value'])
                    # 0-100 → -1 to +1 arasına normalize et
                    return (score - 50) / 50
            return 0.0
        except asyncio.TimeoutError:
            logger.warning("Fear & Greed API request timed out")
            return 0.0
        except Exception as e:
            logger.error(f"Fear & Greed index error: {e}")
            return 0.0

    async def get_macro_market_signal(self) -> MacroMarketSignal:
        """Tüm makro metrikleri toplu olarak hesaplar"""
        # Cache kontrolü ve zaman aşımı
        cache_key = "macro_signal"
        current_time = datetime.now()
        
        if cache_key in self._macro_cache:
            signal, cache_time = self._macro_cache[cache_key]
            if (current_time - cache_time).total_seconds() < self.config.macro_cache_timeout:
                return signal
            # Cache expired, remove it
            del self._macro_cache[cache_key]
        
        # Paralel olarak tüm metrikleri hesapla
        try:
            ssr, netflow, fear_greed = await asyncio.gather(
                self.get_ssr_metric(),
                self.get_netflow_metric(),
                self.get_fear_greed_index(),
                return_exceptions=True
            )
            
            # Exception handling
            ssr = ssr if not isinstance(ssr, Exception) else 0.0
            netflow = netflow if not isinstance(netflow, Exception) else 0.0
            fear_greed = fear_greed if not isinstance(fear_greed, Exception) else 0.0
            
            # ETF flow placeholder - gerçek API bulunana kadar
            etf_flow = 0.0  # Gerçek implementasyon için premium veri kaynağı gerekli
            
            overall_score = (ssr + netflow + fear_greed + etf_flow) / 4
            
            signal = MacroMarketSignal(
                ssr_score=ssr,
                netflow_score=netflow,
                etf_flow_score=etf_flow,
                fear_greed_score=fear_greed,
                overall_score=overall_score,
                confidence=0.7,  # Basit confidence metric
                timestamp=current_time
            )
            
            self._macro_cache[cache_key] = (signal, current_time)
            return signal
            
        except Exception as e:
            logger.error(f"Macro market signal error: {e}")
            # Hata durumunda default signal döndür
            return MacroMarketSignal(timestamp=current_time)

    # -------------------------
    # MEVCUT RISK METRIKLERI 
    # -------------------------
]

    async def compute_atr(self, symbol: str, interval: str = "1h", futures: bool = False) -> float:
        """Average True Range hesaplama (placeholder implementation)"""
        # Gerçek implementasyon burada olacak
        try:
            # Örnek implementasyon
            klines = await self.binance.get_klines(symbol, interval, limit=20, futures=futures)
            if not klines:
                return 0.0
                
            high_prices = [float(k[2]) for k in klines]
            low_prices = [float(k[3]) for k in klines]
            close_prices = [float(k[4]) for k in klines]
            
            true_ranges = []
            for i in range(1, len(klines)):
                high_low = high_prices[i] - low_prices[i]
                high_close = abs(high_prices[i] - close_prices[i-1])
                low_close = abs(low_prices[i] - close_prices[i-1])
                true_ranges.append(max(high_low, high_close, low_close))
            
            return mean(true_ranges) if true_ranges else 0.0
        except Exception as e:
            logger.error(f"ATR calculation error for {symbol}: {e}")
            return 0.0

    async def liquidation_proximity(self, symbol: str, position: Optional[dict]) -> float:
        """Liquidation proximity hesaplama (placeholder implementation)"""
        # Gerçek implementasyon burada olacak
        if not position:
            return 1.0
            
        try:
            # Basit bir örnek implementasyon
            price = await self.binance.get_price(symbol, futures=True)
            entry_price = float(position.get("entryPrice", price))
            leverage = float(position.get("leverage", 1.0))
            
            if leverage <= 0 or entry_price <= 0:
                return 1.0
                
            # Basit bir yakınlık hesaplaması
            price_ratio = min(price / entry_price, entry_price / price)
            safety_margin = 0.1  # 10% safety margin
            
            return max(0.0, min(1.0, (price_ratio - safety_margin) / (1 - safety_margin)))
        except Exception as e:
            logger.error(f"Liquidation proximity calculation error for {symbol}: {e}")
            return 1.0

    async def correlation_risk(self, symbol: str, portfolio_symbols: Sequence[str], 
                              interval: str = "1h", futures: bool = False) -> float:
        """Correlation risk hesaplama (placeholder implementation)"""
        # Gerçek implementasyon burada olacak
        try:
            if len(portfolio_symbols) < 2:
                return 0.0
                
            # Basit bir örnek implementasyon
            # Gerçekte korelasyon matrisi hesaplanacak
            return 0.3  # Örnek değer
        except Exception as e:
            logger.error(f"Correlation risk calculation error for {symbol}: {e}")
            return 0.5  # Conservative default

    async def portfolio_var(self, symbols: List[str], lookback: int = 250, 
                           interval: str = "1h", futures: bool = False) -> float:
        """Portfolio Value at Risk hesaplama (placeholder implementation)"""
        # Gerçek implementasyon burada olacak
        try:
            if not symbols:
                return 0.0
                
            # Basit bir örnek implementasyon
            # Gerçekte tarihsel simülasyon veya parametrik VaR hesaplanacak
            return 0.05  # Örnek değer (5% VaR)
        except Exception as e:
            logger.error(f"Portfolio VaR calculation error: {e}")
            return 0.1  # Conservative default

    async def combined_risk_score(
        self,
        symbol: str,
        *,
        account_positions: Optional[List[dict]] = None,
        portfolio_symbols: Optional[Sequence[str]] = None,
        interval: str = "1h",
        lookback: int = 250,
        futures: bool = False,
        include_macro: bool = True  # Yeni parametre: makro sinyalleri dahil et
    ) -> Dict[str, float]:
        """
        Geliştirilmiş risk skoru - makro piyasa sinyallerini entegre eder.
        """
        try:
            # Mevcut mikro metrikleri hesapla
            price = await self.binance.get_price(symbol, futures=futures)
            atr = await self.compute_atr(symbol, interval=interval, futures=futures)
            vol_metric = min(1.0, (price / atr) / 100.0) if atr > 0 else 0.0
            
            pos = None
            if account_positions:
                for p in account_positions:
                    if p.get("symbol", "").upper() == symbol.upper():
                        pos = p
                        break
            liq_prox = await self.liquidation_proximity(symbol, pos) if pos else 1.0
            corr_penalty = 0.0
            if portfolio_symbols:
                corr_penalty = await self.correlation_risk(symbol, portfolio_symbols, interval=interval, futures=futures)
            var = await self.portfolio_var(list(portfolio_symbols or [symbol]), lookback=lookback, interval=interval, futures=futures)

            # Makro sinyalleri al (isteğe bağlı)
            macro_signal = await self.get_macro_market_signal() if include_macro else None
            
            # Ağırlıkları ayarla
            w_vol = 0.30  # Önceki 0.35
            w_liq = 0.20  # Önceki 0.25  
            w_corr = 0.20  # Aynı
            w_var = 0.20  # Aynı
            w_macro = self.config.macro_weight if include_macro and macro_signal else 0.0
            
            # Mikro metrikleri normalize et
            vol_norm = vol_metric
            liq_norm = liq_prox
            corr_norm = 1.0 - corr_penalty
            var_norm = 1.0 - var
            macro_norm = macro_signal.overall_score if macro_signal else 0.0

            # Ağırlıkları normalize et (toplam 1 olacak şekilde)
            total_weight = w_vol + w_liq + w_corr + w_var + w_macro
            if total_weight > 0:
                w_vol /= total_weight
                w_liq /= total_weight
                w_corr /= total_weight
                w_var /= total_weight
                w_macro /= total_weight
            else:
                # Fallback weights if all are zero
                w_vol = w_liq = w_corr = w_var = w_macro = 0.2

            # Nihai skoru hesapla
            score = (
                w_vol * vol_norm + 
                w_liq * liq_norm + 
                w_corr * corr_norm + 
                w_var * var_norm +
                w_macro * macro_norm
            )
            score = max(0.0, min(1.0, score))
            
            logger.info(f"Geliştirilmiş risk skoru için {symbol} = {score:.3f} (macro: {macro_norm:.3f})")
            
            return {
                "symbol": symbol.upper(),
                "price": float(price),
                "atr": float(atr),
                "vol_metric": float(vol_norm),
                "liquidation_proximity": float(liq_norm),
                "correlation_penalty": float(corr_penalty),
                "portfolio_var": float(var),
                "macro_score": float(macro_norm) if macro_signal else 0.0,
                "macro_confidence": float(macro_signal.confidence) if macro_signal else 0.0,
                "score": float(score),
                "score_without_macro": float(score - w_macro * macro_norm) if macro_signal else float(score),
            }
            
        except Exception as exc:
            logger.exception(f"Geliştirilmiş risk skoru hesaplama hatası {symbol}: {exc}")
            return {
                "symbol": symbol.upper(),
                "price": 0.0,
                "atr": 0.0,
                "vol_metric": 0.0,
                "liquidation_proximity": 0.0,
                "correlation_penalty": 1.0,
                "portfolio_var": 1.0,
                "macro_score": 0.0,
                "macro_confidence": 0.0,
                "score": 0.0,
                "score_without_macro": 0.0,
            }

    # -------------------------
    # YENI OZELLIKLER
    # -------------------------
    async def get_market_regime(self) -> str:
        """
        Piyasa rejimini belirler: BULL, BEAR, veya NEUTRAL
        """
        try:
            macro_signal = await self.get_macro_market_signal()
            if macro_signal.overall_score > 0.3:
                return "BULL"
            elif macro_signal.overall_score < -0.3:
                return "BEAR"
            else:
                return "NEUTRAL"
        except Exception as e:
            logger.error(f"Market regime analysis error: {e}")
            return "NEUTRAL"

    async def adaptive_position_sizing(
        self,
        symbol: str,
        base_fraction: float,
        *,
        risk_budget: float = 0.01,
        account_balance: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Piyasa rejimine göre adaptif pozisyon büyüklüğü önerir.
        """
        market_regime = await self.get_market_regime()
        
        # Piyasa rejimine göre adjustment
        regime_multiplier = {
            "BULL": 1.2,    # Bull piyasada daha agresif
            "NEUTRAL": 1.0, # Normal risk
            "BEAR": 0.6     # Bear piyasada daha korunmacı
        }.get(market_regime, 1.0)
        
        adjusted_fraction = base_fraction * regime_multiplier
        adjusted_fraction = max(0.0, min(1.0, adjusted_fraction))
        
        result = {
            "base_fraction": base_fraction,
            "market_regime": market_regime,
            "regime_multiplier": regime_multiplier,
            "adjusted_fraction": adjusted_fraction,
            "recommendation": "AGGRESSIVE" if regime_multiplier > 1.0 else "CONSERVATIVE" if regime_multiplier < 1.0 else "NEUTRAL"
        }
        
        # Max notional hesaplaması (mevcut logic)
        if account_balance:
            try:
                price = await self.binance.get_price(symbol)
                atr = await self.compute_atr(symbol)
                stop_distance = self.config.k_atr_stop * atr if atr > 0 else price * 0.01
                if stop_distance > 0:
                    max_risk_value = account_balance * risk_budget
                    position_notional = max_risk_value * price / stop_distance
                    result["max_notional"] = position_notional * adjusted_fraction
            except Exception as e:
                logger.error(f"Position sizing calculation error: {e}")
                result["max_notional"] = 0.0
        
        return result

    # -------------------------
    # CLEANUP
    # -------------------------
    async def close(self) -> None:
        """Resource cleanup"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        logger.info("RiskManager resources cleaned up")

    async def __aenter__(self) -> RiskManager:
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        await self.close()

# Usage example ve aiogram entegrasyonu
try:
    from aiogram import Router
    from aiogram.types import Message

    router = Router()

    @router.message(commands=["advanced_risk"])
    async def cmd_advanced_risk(message: Message) -> None:
        """Geliştirilmiş risk analiz komutu"""
        parts = message.text.strip().split()
        if len(parts) < 2:
            await message.answer("Usage: /advanced_risk SYMBOL (e.g. /advanced_risk BTCUSDT)")
            return

        symbol = parts[1].upper()
        try:
            BINANCE = getattr(message.bot, "binance_api", None)
            if BINANCE is None:
                await message.answer("Binance API not configured.")
                return
                
            # Config'i environment'dan al
            config = RiskManagerConfig(
                glassnode_api_key=os.getenv("GLASSNODE_API_KEY"),
                macro_weight=float(os.getenv("MACRO_WEIGHT", DEFAULT_MACRO_WEIGHT))
            )
                
            rm = RiskManager(BINANCE, config)
            summary = await rm.combined_risk_score(
                symbol, 
                portfolio_symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                include_macro=True
            )
            
            # Makro analiz sonuçlarını da getir
            macro = await rm.get_macro_market_signal()
            regime = await rm.get_market_regime()
            
            text = (
                f"🎯 **Gelişmiş Risk Analizi - {summary['symbol']}**\n\n"
                f"📊 **Temel Metrikler:**\n"
                f"• Price: ${summary['price']:,.2f}\n"
                f"• ATR: {summary['atr']:.4f}\n"
                f"• Volatility Score: {summary['vol_metric']:.3f}\n"
                f"• Liquidation Safety: {summary['liquidation_proximity']:.3f}\n"
                f"• Correlation Penalty: {summary['correlation_penalty']:.3f}\n"
                f"• Portfolio VaR: {summary['portfolio_var']:.3f}\n\n"
                f"🌍 **Makro Piyasa:**\n"
                f"• SSR Score: {macro.ssr_score:.3f}\n"
                f"• Netflow Score: {macro.netflow_score:.3f}\n"
                f"• Fear & Greed: {macro.fear_greed_score:.3f}\n"
                f"• Overall Macro: {macro.overall_score:.3f}\n"
                f"• Market Regime: {regime}\n\n"
                f"📈 **Risk Skorları:**\n"
                f"• Micro-Only Score: {summary['score_without_macro']:.3f}\n"
                f"• Final Score: {summary['score']:.3f}\n"
                f"• Confidence: {summary['macro_confidence']:.3f}"
            )
            
            await message.answer(text)
            
        except Exception as exc:
            logger.exception("Advanced risk command error: %s", exc)
            await message.answer("Risk analiz hatası. Loglara bakın.")

except ImportError:
    router = None