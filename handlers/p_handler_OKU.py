"""
#handlers/p_handler.py
Binance API ile coin verilerini çeken /p komutu handler'ı.
Aiogram 3.x Router pattern'ine uygun, async/await yapısında,
type hints + docstring + logging ile geliştirilmiş

Telegram'da /p, /pg, /pl gibi komutlar yazarak Binance API üzerinden anlık coin fiyatları, hacimleri, yükselen/düşen coinler gibi verileri almasını sağlar.
hata durumunda 3 defa yeniden deneme yapar.
çekilen ticker verilerini filtreleyip (örneğin sadece yükselenler, düşenler, özel coinler vs), sıralayıp, düzenleyip kullanıcıya gönderilecek metin haline getirir.
Telegram Komutları:
/p [symbols...] → Seçilen coinlerin veya config SCAN_SYMBOLS listesinin fiyatlarını gösterir /p btc bnb ...
/p → SCAN_SYMBOLS (config içinden veya parametre verilirse)
/p eth , /p eth bnb sol → belirtilen coinlere ait bilgi sorgulaması
- /pg [limit]     → En çok yükselen coinleri listeler
/pg → en çok yükselen coinler (default 20, parametre verilirse değişir)
/pg 30 → en çok yükselen ilk 30 coin,
- /pl [limit]     → En çok düşen coinleri listeler
/pl → en çok düşen coinler (default 20, parametre verilirse değişir)
/pl 30 → en çok düşen 30 coin
- /test_api       → Binance API bağlantı testi yapar
/test_api → Binance API bağlantısını test eder
Rapor formatı (coin adı, değişim %, hacim, fiyat )

Binance API entegrasyonu
utils/binance/binance_a.py içindeki get_or_create_binance_api() kullanılacak.
get_custom_symbols_data, get_top_gainers_with_volume, get_top_losers_with_volume gibi hazır metodlar zaten var.
handler_loader.py → mevcut hali zaten yeterli. p_handler.py otomatik yüklenecek.


Yanıt formatı için tablo düzeni (⚡️ Coin | Değişim | Hacim | Fiyat) ayrı bir yardımcı fonksiyon haline getir → tekrar kullanılabilir olur.

örnek rapor
/p  →(default> SCAN_SYMBOLS=BTCUSDT,ETHUSDT,...)
📈 SCAN_SYMBOLS (Hacme Göre)
⚡️Coin | Değişim | Hacim | Fiyat
1. ETH: -0.52% | $627.4M | 4477.12
2. BTC: -0.59% | $550.0M | 115348.12
3. BNB: 1.82% | $543.8M | 1043.19
4. SOL: -1.42% | $315.5M | 237.05
5. TRX: -1.35% | $67.3M | 0.3432
6. SUI: -2.00% | $66.3M | 3.6124
7. PEPE: -2.78% | $55.1M | 1.048e-05
8. CAKE: -1.99% | $54.7M | 2.906
9. ARPA: 2.42% | $2.6M | 0.02373
10. TURBO: -3.19% | $2.0M | 0.004013


/p btc eth bnb sol

📈 Seçili Coinler
⚡Coin | Değişim | Hacim | Fiyat
1. BNB: 1.79% | $543.8M | 1042.1
2. ETH: -0.56% | $627.9M | 4474.3
3. BTC: -0.65% | $550.7M | 115272.43
4. SOL: -1.37% | $315.2M | 236.91

"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime

from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.utils.markdown import code

from config import get_config
from utils.binance.binance_a import get_or_create_binance_api

logger = logging.getLogger(__name__)
router = Router(name="p_handler")

# Global instance - Artık get_or_create_binance_api kullanıyoruz
_binance_instance = None

async def get_binance() -> BinanceAPI:
    """BinanceAPI instance'ını döndürür."""
    global _binance_instance
    if _binance_instance is None:
        try:
            config = await get_config()
            
            _binance_instance = await get_or_create_binance_api(
                api_key=config.BINANCE_API_KEY,
                api_secret=config.BINANCE_API_SECRET,
                cache_ttl=30,
                base_url=config.BINANCE_BASE_URL,
                fapi_url=config.BINANCE_FAPI_URL,
                failure_threshold=config.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                reset_timeout=config.CIRCUIT_BREAKER_RESET_TIMEOUT
            )
            
            logger.info("✅ BinanceAPI instance created for p_handler")
            
        except Exception as e:
            logger.error(f"❌ BinanceAPI instance oluşturulamadı: {e}", exc_info=True)
            raise
    
    return _binance_instance

# Gereksiz fonksiyonları kaldır, doğrudan binance_a metodlarını kullan
async def fetch_tickers_with_retry(max_retries: int = 3) -> List[Dict[str, Any]]:
    """Retry mekanizmalı ticker veri çekme."""
    for attempt in range(max_retries):
        try:
            binance = await get_binance()
            tickers = await binance.get_all_24h_tickers()
            
            if tickers:
                logger.info(f"✅ {attempt + 1}. denemede veri alındı: {len(tickers)} ticker")
                return tickers
                
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"❌ {attempt + 1}. deneme başarısız: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
    
    logger.error(f"❌ Tüm {max_retries} deneme başarısız oldu")
    return []

def format_volume(volume: float) -> str:
    """Hacim formatlama."""
    if volume >= 1_000_000_000:
        return f"${volume/1_000_000_000:.1f}B"
    elif volume >= 1_000_000:
        return f"${volume/1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"${volume/1_000:.1f}K"
    else:
        return f"${volume:.0f}"

def format_price(price: float) -> str:
    """Fiyat formatlama."""
    if price >= 1000:
        return f"{price:,.0f}"
    elif price >= 1:
        return f"{price:.2f}"
    elif price >= 0.01:
        return f"{price:.4f}"
    else:
        return f"{price:.6f}"

def format_percentage(change: float) -> str:
    """Yüzde formatlama."""
    return f"{change:+.2f}%"

async def generate_price_message(mode: str = "default", limit: int = 20, 
                               custom_symbols: Optional[List[str]] = None) -> str:
    """Fiyat mesajını oluşturur."""
    try:
        binance = await get_binance()
        
        if mode == "default":
            config = await get_config()
            tickers = await binance.get_custom_symbols_data(config.SCAN_SYMBOLS)
            tickers.sort(key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
            title = "📈 SCAN_SYMBOLS (Hacme Göre)"
            
        elif mode == "gainers":
            tickers = await binance.get_top_gainers_with_volume(limit=limit, min_volume_usdt=1000000)
            title = f"📈 En Çok Yükselen {len(tickers)} Coin (Min. $1M Hacim)"
            
        elif mode == "losers":
            tickers = await binance.get_top_losers_with_volume(limit=limit, min_volume_usdt=1000000)
            title = f"📉 En Çok Düşen {len(tickers)} Coin (Min. $1M Hacim)"
            
        elif mode == "custom" and custom_symbols:
            tickers = await binance.get_custom_symbols_data(custom_symbols)
            tickers.sort(key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
            title = "📈 Seçili Coinler"
            
        else:
            return "❌ Geçersiz mod!"
        
        if not tickers:
            return "❌ Eşleşen coin bulunamadı."
        
        # Mesajı oluştur
        lines = [title, "⚡Coin | Değişim | Hacim | Fiyat"]
        
        for i, ticker in enumerate(tickers[:limit], 1):
            symbol = ticker.get('symbol', 'N/A')
            change_percent = float(ticker.get('priceChangePercent', 0))
            volume = float(ticker.get('quoteVolume', 0))
            price = float(ticker.get('lastPrice', 0))
            
            display_symbol = symbol.replace('USDT', '') if symbol.endswith('USDT') else symbol
            
            line = (f"{i}. {display_symbol}: "
                    f"{format_percentage(change_percent)} | "
                    f"{format_volume(volume)} | "
                    f"{format_price(price)}")
            lines.append(line)
        
        lines.append(f"\n🕒 Son güncelleme: {datetime.now().strftime('%H:%M:%S')}")
        return "\n".join(lines)
        
    except Exception as e:
        logger.error(f"❌ Mesaj oluşturulamadı: {e}")
        return "❌ Veri işlenirken hata oluştu."

@router.message(Command("p"))
async def p_command_handler(message: Message):
    """Ana /p komutu handler'ı."""
    try:
        args = message.text.split()[1:]
        
        if not args:
            response = await generate_price_message(mode="default")
            
        elif args[0].isdigit():
            limit = min(int(args[0]), 50)
            response = await generate_price_message(mode="gainers", limit=limit)
            
        elif args[0].lower() == 'd':
            limit = min(int(args[1]), 50) if len(args) > 1 and args[1].isdigit() else 20
            response = await generate_price_message(mode="losers", limit=limit)
            
        else:
            custom_symbols = [f"{arg.upper()}USDT" for arg in args if arg.strip()]
            response = await generate_price_message(mode="custom", custom_symbols=custom_symbols)
        
        await message.answer(code(response))
        
    except Exception as e:
        logger.error(f"❌ /p komutu hatası: {e}", exc_info=True)
        await message.answer("❌ Bir hata oluştu. Lütfen daha sonra tekrar deneyin.")

# Diğer komutlar için benzer optimizasyonlar...