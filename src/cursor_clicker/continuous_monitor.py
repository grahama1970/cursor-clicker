"""
Continuous screenshot monitoring for Cursor.sh.

This module provides a standalone script that continuously captures screenshots
of the Cursor.sh window at regular intervals, without loading the ML model.
"""

import os
import time
import logging
import argparse
from cursor_clicker.screenshot_manager import capture_and_save

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("continuous_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def continuous_monitor(interval=5, use_timestamp=False, compress=True, max_width=800, quality=75):
    """
    Continuously capture screenshots at regular intervals.
    
    Args:
        interval (int): Time between screenshots in seconds
        use_timestamp (bool): Whether to use timestamp in filename or overwrite
        compress (bool): Whether to compress screenshots
        max_width (int): Maximum width for compression
        quality (int): JPEG quality for compression
    """
    logger.info(f"Starting continuous monitor with {'timestamp' if use_timestamp else 'overwrite'} mode")
    logger.info(f"Screenshot interval: {interval} seconds")
    logger.info(f"Compression settings: enabled={compress}, max_width={max_width}, quality={quality}")
    
    screenshot_dir = "screenshots"
    
    try:
        count = 0
        start_time = time.time()
        
        while True:
            count += 1
            logger.info(f"Capturing screenshot #{count}...")
            
            screenshot_path = capture_and_save(
                directory=screenshot_dir,
                use_timestamp=use_timestamp,
                compress=compress,
                max_width=max_width,
                quality=quality
            )
            
            if screenshot_path:
                logger.info(f"Screenshot saved to: {screenshot_path}")
                
                # Log storage statistics every 10 screenshots if using timestamp mode
                if use_timestamp and count % 10 == 0:
                    total_size = 0
                    file_count = 0
                    
                    for filename in os.listdir(screenshot_dir):
                        file_path = os.path.join(screenshot_dir, filename)
                        total_size += os.path.getsize(file_path)
                        file_count += 1
                    
                    elapsed_time = time.time() - start_time
                    avg_size = total_size / file_count if file_count > 0 else 0
                    
                    logger.info(f"Storage stats after {count} screenshots:")
                    logger.info(f"  Total files: {file_count}")
                    logger.info(f"  Total size: {total_size / (1024*1024):.2f} MB")
                    logger.info(f"  Average size: {avg_size / 1024:.2f} KB per file")
                    logger.info(f"  Running for: {elapsed_time / 60:.2f} minutes")
                    logger.info(f"  Storage rate: {total_size / (1024*1024) / (elapsed_time / 3600):.2f} MB/hour")
            else:
                logger.error("Failed to capture screenshot")
            
            # Wait for next interval
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("Monitor stopped by user")
    except Exception as e:
        logger.exception(f"Monitor stopped due to error: {e}")

def main():
    """
    Run the continuous monitor with command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Continuous screenshot monitoring for Cursor.sh")
    parser.add_argument("--interval", type=int, default=5, help="Seconds between screenshots")
    parser.add_argument("--timestamp", action="store_true", help="Use timestamp in filenames instead of overwriting")
    parser.add_argument("--no-compress", action="store_true", help="Disable compression")
    parser.add_argument("--width", type=int, default=800, help="Maximum width for compression")
    parser.add_argument("--quality", type=int, default=75, help="JPEG quality (1-100)")
    
    args = parser.parse_args()
    
    continuous_monitor(
        interval=args.interval,
        use_timestamp=args.timestamp,
        compress=not args.no_compress,
        max_width=args.width,
        quality=args.quality
    )

if __name__ == "__main__":
    main() 