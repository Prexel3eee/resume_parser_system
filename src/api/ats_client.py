import httpx
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ATSAPIClient:
    """Async ATS API client with retry logic"""
    
    def __init__(self, 
                 api_url: str,
                 api_key: str,
                 batch_size: int = 100,
                 max_retries: int = 3):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.batch_size = batch_size
        self.max_retries = max_retries
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def push_batch(self, 
                        resumes: List[Dict],
                        session: httpx.AsyncClient) -> Dict:
        """Push batch of resumes to API"""
        response = await session.post(
            f"{self.api_url}/resumes/bulk",
            json={"resumes": resumes},
            headers=self.headers,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()
        
    async def push_all_resumes(self, resumes: List[Dict]) -> Dict:
        """Push all resumes in batches"""
        results = {
            "success": 0,
            "failed": 0,
            "errors": []
        }
        
        async with httpx.AsyncClient() as session:
            # Process in batches
            for i in range(0, len(resumes), self.batch_size):
                batch = resumes[i:i + self.batch_size]
                
                try:
                    response = await self.push_batch(batch, session)
                    results["success"] += response.get("processed", 0)
                    
                    logger.info(
                        f"Pushed batch {i//self.batch_size + 1}: "
                        f"{response.get('processed')} resumes"
                    )
                    
                except Exception as e:
                    results["failed"] += len(batch)
                    results["errors"].append({
                        "batch": i//self.batch_size + 1,
                        "error": str(e)
                    })
                    logger.error(f"Failed to push batch: {e}")
                    
                # Rate limiting
                await asyncio.sleep(1)
                
        return results 