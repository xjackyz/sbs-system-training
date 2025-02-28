"""
Redis缓存管理器
实现多级缓存和智能预热
"""

import redis
import json
from typing import Dict, Any, Optional, List
from ..utils.logger import setup_logger

logger = setup_logger('redis_manager')

class RedisManager:
    """Redis缓存管理器"""
    
    def __init__(self, config: Dict):
        """
        初始化Redis管理器
        
        Args:
            config: Redis配置
        """
        self.config = config
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            password=config.get('redis_password'),
            decode_responses=True
        )
        
        # 多级缓存配置
        self.l1_cache = {}  # 内存缓存
        self.l1_capacity = config.get('l1_capacity', 1000)
        
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            缓存数据
        """
        try:
            # 检查L1缓存
            if key in self.l1_cache:
                logger.debug(f"L1缓存命中: {key}")
                return self.l1_cache[key]
                
            # 检查Redis缓存
            data = self.redis_client.get(key)
            if data:
                logger.debug(f"Redis缓存命中: {key}")
                value = json.loads(data)
                # 更新L1缓存
                self._update_l1_cache(key, value)
                return value
                
            return None
            
        except Exception as e:
            logger.error(f"获取缓存失败: {str(e)}")
            return None
            
    def set(self, key: str, value: Any, expire: int = None) -> bool:
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            value: 缓存值
            expire: 过期时间（秒）
            
        Returns:
            是否成功
        """
        try:
            # 更新L1缓存
            self._update_l1_cache(key, value)
            
            # 更新Redis缓存
            data = json.dumps(value)
            if expire:
                self.redis_client.setex(key, expire, data)
            else:
                self.redis_client.set(key, data)
                
            return True
            
        except Exception as e:
            logger.error(f"设置缓存失败: {str(e)}")
            return False
            
    def _update_l1_cache(self, key: str, value: Any) -> None:
        """
        更新L1缓存
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        if len(self.l1_cache) >= self.l1_capacity:
            # LRU策略：移除最早的项
            self.l1_cache.pop(next(iter(self.l1_cache)))
        self.l1_cache[key] = value
        
    def preload_data(self, keys: List[str]) -> None:
        """
        预热缓存数据
        
        Args:
            keys: 需要预热的键列表
        """
        try:
            # 批量获取数据
            pipe = self.redis_client.pipeline()
            for key in keys:
                pipe.get(key)
            values = pipe.execute()
            
            # 更新L1缓存
            for key, value in zip(keys, values):
                if value:
                    self._update_l1_cache(key, json.loads(value))
                    
            logger.info(f"缓存预热完成，共 {len(keys)} 个键")
            
        except Exception as e:
            logger.error(f"缓存预热失败: {str(e)}")
            
    def clear_cache(self, pattern: str = None) -> None:
        """
        清理缓存
        
        Args:
            pattern: 匹配模式
        """
        try:
            if pattern:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                self.redis_client.flushall()
                
            # 清理L1缓存
            self.l1_cache.clear()
            
            logger.info("缓存已清理")
            
        except Exception as e:
            logger.error(f"清理缓存失败: {str(e)}")
            
    def get_stats(self) -> Dict:
        """
        获取缓存统计信息
        
        Returns:
            统计信息
        """
        try:
            info = self.redis_client.info()
            return {
                'used_memory': info['used_memory_human'],
                'connected_clients': info['connected_clients'],
                'total_keys': len(self.redis_client.keys('*')),
                'l1_cache_size': len(self.l1_cache)
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {} 