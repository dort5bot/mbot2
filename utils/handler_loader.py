# utils/handler_loader.py
"""
KOD2 > daha nitelikli
Type hints eklenmiş (Dict[str, int]) + Daha okunaklı ve sade
*_handler.py pattern'i ile sadece handler dosyalarını yüklüyor+daha profesyonel
    Eğer farklı pattern'lerde dosyaları da yüklemek isterseniz, glob pattern'ini değiştirebilirsiniz. Örneğin:
    *.py: Tüm Python dosyaları
    *_*.py: Alt çizgi içeren tüm dosyalar
    *[!__init__].py: init hariç tüm dosyalar
    # handler_loader.py'de pattern'i genişletmek için
patterns = ["*_handler.py", "analysis*.py", "*.py"]  # Daha esnek

import_module daha güvenli ve standart+Manuel spec oluşturmaya gerek yok
Daha basit ve etkili cache temizleme+Exception handling daha iyi

"""
import logging
from importlib import import_module
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

async def load_handlers(dispatcher) -> Dict[str, int]:
    """
    Tüm handler dosyalarını yükler ve Dispatcher'a ekler.
    Her handler dosyasında `router = Router()` olmalı.
    """
    handlers_path = Path("handlers")
    loaded = 0
    failed = 0
    
      # ⬇️ Genişletilmiş dosya eşleme kuralları
    patterns = ["*_handler.py", "analysis*.py"]  # örnek: analysis_a.py gibi dosyaları da dahil eder

    for pattern in patterns:
        for file in handlers_path.glob(pattern):
            module_name = f"handlers.{file.stem}"
            try:
                module = import_module(module_name)
                if hasattr(module, "router"):
                    dispatcher.include_router(module.router)
                    logger.info(f"✅ Handler yüklendi: {file.stem}")
                    loaded += 1
                else:
                    logger.warning(f"⚠️ Router bulunamadı: {file.stem}")
                    failed += 1
            except Exception as e:
                logger.error(f"❌ Handler yüklenemedi: {file.stem} -> {e}")
                failed += 1

    return {"loaded": loaded, "failed": failed}

def clear_handler_cache():
    """Handler modüllerinin cache’ini temizler (opsiyonel, hot-reload için)."""
    import sys
    for key in list(sys.modules.keys()):
        if key.startswith("handlers."):
            del sys.modules[key]
