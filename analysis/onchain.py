"""
On-chain Analiz Modülü - Güncellenmiş Singleton Pattern

Bu modül, Stablecoin Supply Ratio, Exchange Net Flow ve ETF Flows gibi
ana on-chain metrikleri hesaplar. 
Çıktılar -1 (Bearish) ile +1 (Bullish) arasında normalize edilir.

Güncellemeler:
- Singleton pattern düzeltildi
- BinanceAPI parametre olarak alınıyor
- Daha iyi error handling
- Config entegrasyonu için hazır
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional
import aiohttp
import numpy as np

from utils.binance.binance_a import BinanceAPI
from config import ONCHAIN_CONFIG

logger = logging.getLogger(__name__)

# analysis/onchain.py başlangıcına config import ekleyin
from config import ONCHAIN_CONFIG

# Sınıf içinde config kullanımı
class OnchainAnalyzer:
    def __init__(self, binance_api: Optional[BinanceAPI] = None, 
                 config: Optional[Dict[str, Any]] = None) -> None:
        """
        OnchainAnalyzer initialization with config.
        
        Args:
            binance_api: BinanceAPI instance
            config: Configuration dictionary
        """
        if not self._initialized:
            self.binance = binance_api
            self.config = config or ONCHAIN_CONFIG  # Default config kullan
            self.session: Optional[aiohttp.ClientSession] = None
            self._cache: Dict[str, Any] = {}
            self._cache_timestamps: Dict[str, float] = {}
            self._initialized = True
            logger.info("OnchainAnalyzer initialized with config")

    async def _get_cached_data(self, key: str, ttl: int) -> Optional[Any]:
        """Cache'ten veri al"""
        current_time = time.time()
        if key in self._cache and current_time - self._cache_timestamps[key] < ttl:
            return self._cache[key]
        return None

    async def _set_cached_data(self, key: str, data: Any) -> None:
        """Cache'e veri kaydet"""
        self._cache[key] = data
        self._cache_timestamps[key] = time.time()

    async def stablecoin_supply_ratio(self) -> float:
        """Config ile güncellenmiş SSR metodu"""
        try:
            # Cache kontrolü
            cache_key = "ssr_data"
            cache_ttl = self.config["CACHE_TTL"]["ssr"]
            cached_data = await self._get_cached_data(cache_key, cache_ttl)
            
            if cached_data is not None:
                return cached_data
            
            if not self.binance:
                logger.error("BinanceAPI not set for OnchainAnalyzer")
                return self.config["FALLBACK_VALUES"]["ssr"]
            
            # Config'ten threshold'ları al
            ssr_thresholds = self.config["SSR_THRESHOLDS"]
            
            # Glassnode API'den veri al
            ssr_data = await self._get_glassnode_data("metrics/indicators/ssr", {
                'a': 'BTC',
                'i': '24h',
                's': int(time.time()) - 86400,
                'u': int(time.time())
            })
            
            if ssr_data and len(ssr_data) > 0:
                latest_ssr = ssr_data[-1]['v']
                
                # Config'ten alınan threshold'lara göre normalize et
                if latest_ssr > ssr_thresholds["bearish"]:
                    result = -1.0
                elif latest_ssr < ssr_thresholds["bullish"]:
                    result = 1.0
                else:
                    # Linear normalization
                    result = (ssr_thresholds["neutral"] - latest_ssr) / (
                        ssr_thresholds["bearish"] - ssr_thresholds["bullish"]) * 2
                
                # Cache'e kaydet
                await self._set_cached_data(cache_key, result)
                return result
                
            else:
                # Fallback hesaplama
                btc_data = await self.binance.get_price("BTCUSDT")
                usdt_market_cap = 80_000_000_000
                btc_market_cap = btc_data * 19_500_000
                
                ssr = btc_market_cap / usdt_market_cap
                normalized_ssr = max(-1.0, min(1.0, 
                    (ssr_thresholds["neutral"] - ssr) / (
                        ssr_thresholds["bearish"] - ssr_thresholds["bullish"]) * 2))
                
                logger.debug(f"SSR fallback: {ssr:.2f}, normalize: {normalized_ssr:.3f}")
                
                # Cache'e kaydet
                await self._set_cached_data(cache_key, normalized_ssr)
                return normalized_ssr
                
        except Exception as e:
            logger.error(f"SSR hesaplanırken hata: {e}")
            return self.config["FALLBACK_VALUES"]["ssr"]

    async def aggregate_score(self) -> Dict[str, Any]:
        """Config ile güncellenmiş aggregate score"""
        try:
            # Config'ten ağırlıkları al
            weights = self.config["METRIC_WEIGHTS"]
            
            # Tüm metrikleri paralel çalıştır
            tasks = {
                "stablecoin_supply_ratio": self.stablecoin_supply_ratio(),
                "exchange_net_flow": self.exchange_net_flow(),
                "etf_flows": self.etf_flows(),
                "fear_greed_index": self.fear_greed_index()
            }
            
            results = {}
            for name, task in tasks.items():
                try:
                    results[name] = await asyncio.wait_for(
                        task, 
                        timeout=self.config["API_TIMEOUTS"].get(name.split('_')[0], 30)
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"{name} timeout, fallback değer kullanılıyor")
                    results[name] = self.config["FALLBACK_VALUES"].get(name, 0.0)
                except Exception as e:
                    logger.error(f"{name} hesaplanırken hata: {e}")
                    results[name] = self.config["FALLBACK_VALUES"].get(name, 0.0)

            # Ağırlıklı ortalama
            total_score = sum(results[name] * weights.get(name, 0.25) for name in results)
            total_score = round(total_score, 3)

            result = {
                **results,
                "aggregate": total_score,
            }
            
            logger.info(f"On-chain analiz sonucu: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Aggregate score hesaplanırken beklenmeyen hata: {e}")
            # Config'ten fallback değerleri kullan
            fallback_values = self.config["FALLBACK_VALUES"]
            return {
                "stablecoin_supply_ratio": fallback_values["ssr"],
                "exchange_net_flow": fallback_values["netflow"],
                "etf_flows": fallback_values["etf_flows"],
                "fear_greed_index": fallback_values["fear_greed"],
                "aggregate": 0.0,
            }

# Aiogram 3.x Router entegrasyonu
from aiogram import Router, F
from aiogram.types import Message

router = Router()

@router.message(F.text.lower() == "onchain")
async def onchain_handler(message: Message) -> None:
    """
    Telegram bot için On-chain analiz handler.
    """
    try:
        # Bot'tan analyzer instance'ını al
        analyzer = getattr(message.bot, 'onchain_analyzer', None)
        if analyzer is None:
            # Eğer yoksa oluştur
            binance_api = getattr(message.bot, 'binance_api', None)
            if not binance_api:
                await message.answer("Binance API bağlantısı kurulamadı")
                return
            
            analyzer = get_onchain_analyzer(binance_api)
            message.bot.onchain_analyzer = analyzer
        
        result = await analyzer.aggregate_score()

        text = (
            "🔗 On-Chain Analiz:\n"
            f"- Stablecoin Supply Ratio: {result['stablecoin_supply_ratio']:.3f}\n"
            f"- Exchange Net Flow: {result['exchange_net_flow']:.3f}\n"
            f"- ETF Flows: {result['etf_flows']:.3f}\n"
            f"- Fear & Greed: {result['fear_greed_index']:.3f}\n"
            f"📊 Genel Skor: {result['aggregate']:.3f}"
        )

        await message.answer(text)
        
    except Exception as e:
        logger.error(f"Onchain handler error: {e}")
        await message.answer("On-chain analiz sırasında hata oluştu")

# Singleton instance getter
def get_onchain_analyzer(binance_api: Optional[BinanceAPI] = None, 
                         config: Optional[Dict[str, Any]] = None) -> OnchainAnalyzer:
    """
    OnchainAnalyzer singleton instance'ını döndürür.
    
    Args:
        binance_api: BinanceAPI instance (opsiyonel)
        config: Konfigürasyon ayarları (opsiyonel)
        
    Returns:
        OnchainAnalyzer instance
    """
    return OnchainAnalyzer(binance_api, config)

# Test için
async def test_onchain_analyzer():
    """Test function"""
    from utils.binance.binance_request import BinanceHTTPClient
    from utils.binance.binance_circuit_breaker import CircuitBreaker
    
    # Mock veya gerçek BinanceAPI
    http_client = BinanceHTTPClient(api_key="test", secret_key="test")
    cb = CircuitBreaker()
    binance_api = BinanceAPI(http_client, cb)
    
    async with get_onchain_analyzer(binance_api) as analyzer:
        result = await analyzer.aggregate_score()
        print("On-chain analysis result:", result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_onchain_analyzer())