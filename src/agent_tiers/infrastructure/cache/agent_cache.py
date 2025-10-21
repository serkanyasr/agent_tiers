import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pydantic_ai import Agent
import logging
from threading import RLock

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    agent: Agent
    created_at: float
    last_accessed: float
    access_count: int

class AgentCache:
    """Enhanced agent cache with TTL, size limits, and thread safety"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self._agents: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = RLock()  # Thread safety
        self._cleanup_task: Optional[asyncio.Task] = None
        self._instance = None
        
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    def get(self, session_id: str) -> Optional[Agent]:
        """Get agent for session with access tracking"""
        with self._lock:
            entry = self._agents.get(session_id)
            if entry:
                current_time = time.time()
                # Check TTL
                if current_time - entry.created_at > self._ttl_seconds:
                    del self._agents[session_id]
                    logger.info(f"Agent expired for session {session_id}")
                    return None
                
                # Update access info
                entry.last_accessed = current_time
                entry.access_count += 1
                return entry.agent
            return None
    
    def set(self, session_id: str, agent: Agent):
        """Set agent for session with size management"""
        with self._lock:
            current_time = time.time()
            
            # Check if we need to evict
            if len(self._agents) >= self._max_size and session_id not in self._agents:
                self._evict_lru()
            
            self._agents[session_id] = CacheEntry(
                agent=agent,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1
            )
            logger.info(f"Cached agent for session {session_id}")
    
    def remove(self, session_id: str):
        """Remove agent for session"""
        with self._lock:
            if session_id in self._agents:
                del self._agents[session_id]
                logger.info(f"Removed agent for session {session_id}")
    
    def clear(self):
        """Clear all cached agents"""
        with self._lock:
            self._agents.clear()
            logger.info("Cleared all cached agents")
    
    def _evict_lru(self):
        """Evict least recently used agent"""
        if not self._agents:
            return
        
        lru_session_id = min(
            self._agents.keys(),
            key=lambda sid: self._agents[sid].last_accessed
        )
        del self._agents[lru_session_id]
        logger.info(f"Evicted LRU agent for session {lru_session_id}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of expired entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Clean every 5 minutes
                current_time = time.time()
                
                with self._lock:
                    expired_sessions = [
                        sid for sid, entry in self._agents.items()
                        if current_time - entry.created_at > self._ttl_seconds
                    ]
                    
                    for sid in expired_sessions:
                        del self._agents[sid]
                    
                    if expired_sessions:
                        logger.info(f"Cleaned up {len(expired_sessions)} expired agents")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "total_agents": len(self._agents),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
                "memory_usage_estimate": sum(
                    1 for entry in self._agents.values()
                    if entry.access_count > 0
                )
            }

# Singleton pattern for backward compatibility
def get_agent_cache(max_size: int = 1000, ttl_seconds: int = 3600) -> AgentCache:
    """Get singleton agent cache instance"""
    if not hasattr(get_agent_cache, '_instance') or get_agent_cache._instance is None:
        get_agent_cache._instance = AgentCache(max_size, ttl_seconds)
    return get_agent_cache._instance