import os
import logging
import asyncio
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("kairos.storage")

class R2StorageManager:
    def __init__(self):
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL")
        self.access_key_id = os.getenv("S3_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.data_dir = os.getenv("DATA_DIR", "./data")
        
        self.enabled = all([self.endpoint_url, self.access_key_id, self.secret_access_key, self.bucket_name])
        
        if self.enabled:
            # Initialize the S3 client wrapper targeting Cloudflare R2
            self.client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key
            )
        else:
            logger.warning("Cloudflare R2 environment variables are incomplete. Cloud backup sync is disabled.")

    def _upload_worker(self, local_path: str, cloud_key: str):
        """Synchronous task runner execution targeting worker threads."""
        try:
            self.client.upload_file(local_path, self.bucket_name, cloud_key)
            logger.info(f"Successfully uploaded {cloud_key} to Cloudflare R2.")
        except ClientError as e:
            logger.error(f"Failed uploading {cloud_key} to R2: {e}")

    def _download_worker(self, cloud_key: str, local_path: str) -> bool:
        """Synchronous file downloading routine."""
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.client.download_file(self.bucket_name, cloud_key, local_path)
            logger.info(f"Successfully downloaded {cloud_key} from Cloudflare R2.")
            return True
        except ClientError as e:
            # Catch 404 meaning the file doesn't exist yet on the very first fresh deploy
            if e.response['Error']['Code'] == "404":
                logger.warning(f"File {cloud_key} not found in bucket. Will create locally.")
            else:
                logger.error(f"Error downloading {cloud_key} from R2: {e}")
            return False

    async def sync_down(self, files: list[str]):
        """Runs at main.py startup to download state before services begin."""
        if not self.enabled:
            return
        
        logger.info("Initializing baseline data fetch sequence...")
        for filename in files:
            local_path = os.path.join(self.data_dir, filename)
            # Offload blocking I/O calls onto threadpool executors safely
            await asyncio.to_thread(self._download_worker, filename, local_path)

    async def sync_up_background(self, filename: str):
        """Non-blocking background push invoked instantly upon writebacks."""
        if not self.enabled:
            return
        
        local_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(local_path):
            logger.error(f"Local target path {local_path} does not exist to sync up.")
            return

        # Fire and forget onto background loop thread
        asyncio.create_task(asyncio.to_thread(self._upload_worker, local_path, filename))
    
    def _list_and_download_sessions_worker(self):
        """Worker thread to download all past active session history files."""
        try:
            # List objects under the sessions/ folder inside your R2 bucket
            response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix="sessions/")
            if "Contents" not in response:
                return
            
            for obj in response["Contents"]:
                cloud_key = obj["Key"]
                if cloud_key.endswith(".json"):
                    local_path = os.path.join(self.data_dir, cloud_key)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    self.client.download_file(self.bucket_name, cloud_key, local_path)
            logger.info("Successfully synchronized all cloud session histories locally.")
        except Exception as e:
            logger.error(f"Error restoring session histories from R2: {e}")

    async def sync_down_sessions(self):
        """Pulls down all past conversation sessions at startup before channels open."""
        if not self.enabled:
            return
        await asyncio.to_thread(self._list_and_download_sessions_worker)



# Global instantiation handle
storage_manager = R2StorageManager()