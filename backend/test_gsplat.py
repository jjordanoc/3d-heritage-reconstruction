"""
Test script for Gaussian Splatting Training Pipeline

Usage:
    python test_gsplat.py --scene_id mijato --base_url http://your-app.modal.run --wait
"""

import requests
import time
import argparse
import json


def trigger_gsplat_training(
    base_url: str,
    scene_id: str,
    data_factor: int = 1,
    max_steps: int = 30000
) -> dict:
    """Trigger gsplat training."""
    url = f"{base_url}/scene/{scene_id}/train_gsplat"
    
    print(f"🚀 Triggering gsplat training for scene: {scene_id}")
    print(f"   URL: {url}")
    print(f"   data_factor: {data_factor}")
    print(f"   max_steps: {max_steps}")
    
    params = {
        "data_factor": data_factor,
        "max_steps": max_steps
    }
    
    response = requests.post(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    print(f"✅ Response: {json.dumps(result, indent=2)}")
    
    return result


def check_gsplat_status(base_url: str, scene_id: str) -> dict:
    """Check gsplat training status."""
    url = f"{base_url}/scene/{scene_id}/train_gsplat/status"
    
    response = requests.get(url)
    response.raise_for_status()
    
    return response.json()


def download_model(base_url: str, scene_id: str, output_path: str = None):
    """Download trained .ply model."""
    url = f"{base_url}/scene/{scene_id}/gsplat/model"
    
    print(f"📥 Downloading model from: {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    if output_path is None:
        output_path = f"{scene_id}_gaussian_splatting.ply"
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"✅ Model saved to: {output_path}")
    return output_path


def wait_for_completion(
    base_url: str,
    scene_id: str,
    timeout: int = 1800,
    poll_interval: int = 15
):
    """
    Wait for gsplat training to complete.
    
    Args:
        base_url: Base URL of the API
        scene_id: Scene ID
        timeout: Maximum time to wait in seconds (default: 30 min)
        poll_interval: Time between status checks in seconds
    """
    print(f"\n⏳ Waiting for training to complete (timeout: {timeout}s)...")
    
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > timeout:
            print(f"❌ Timeout after {timeout}s")
            return None
        
        status_result = check_gsplat_status(base_url, scene_id)
        status = status_result.get('status')
        
        if status == 'complete':
            print(f"\n✅ Training complete after {elapsed:.1f}s")
            print(f"\nResults:")
            print(json.dumps(status_result, indent=2))
            return status_result
        
        elif status == 'processing':
            progress = status_result.get('progress', 'N/A')
            print(f"   [{elapsed:.1f}s] Training... {progress}")
        
        elif status == 'failed':
            print(f"\n❌ Training failed:")
            print(status_result.get('error', 'Unknown error'))
            return None
        
        elif status == 'not_started':
            print(f"   [{elapsed:.1f}s] Not started yet...")
        
        else:
            print(f"   [{elapsed:.1f}s] Unknown status: {status}")
        
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(
        description="Test Gaussian Splatting Training Pipeline"
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        required=True,
        help="Scene ID to process (e.g., 'mijato')"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        required=True,
        help="Base URL of the API (e.g., 'http://your-app.modal.run')"
    )
    parser.add_argument(
        "--data_factor",
        type=int,
        default=1,
        choices=[1, 2, 4],
        help="Image downscale factor (1=full, 2=half, 4=quarter)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=30000,
        help="Training iterations (default: 30000)"
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Only check status without triggering training"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for training to complete"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the trained model after completion"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for downloaded model"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout in seconds when waiting (default: 1800 = 30 min)"
    )
    
    args = parser.parse_args()
    
    # Remove trailing slash from base_url
    base_url = args.base_url.rstrip('/')
    
    try:
        if not args.check_only:
            # Trigger training
            trigger_result = trigger_gsplat_training(
                base_url,
                args.scene_id,
                args.data_factor,
                args.max_steps
            )
            print(f"\n📋 Check status at: {base_url}{trigger_result.get('check_status_at', '')}")
        
        if args.wait or args.check_only:
            # Wait for completion or just check status
            if args.wait and not args.check_only:
                print("\nWaiting a few seconds for training to start...")
                time.sleep(5)
            
            result = wait_for_completion(
                base_url,
                args.scene_id,
                timeout=args.timeout,
                poll_interval=15
            )
            
            if result and result.get('status') == 'complete':
                # Print summary
                print("\n" + "="*80)
                print("SUMMARY")
                print("="*80)
                
                print(f"\nScene ID: {args.scene_id}")
                print(f"Status: {result.get('status')}")
                print(f"Model size: {result.get('model_size_mb', 'N/A')} MB")
                print(f"Download URL: {base_url}{result.get('download_url', '')}")
                
                config = result.get('config', {})
                if config:
                    print(f"\nTraining config:")
                    print(f"  Data factor: {config.get('data_factor', 'N/A')}")
                    print(f"  Max steps: {config.get('max_steps', 'N/A')}")
                
                print("="*80)
                
                # Download if requested
                if args.download:
                    print("\n")
                    download_model(base_url, args.scene_id, args.output)
        
        else:
            # Just triggered, not waiting
            print("\n✅ Training started. Check status with:")
            print(f"   python {__file__} --scene_id {args.scene_id} --base_url {base_url} --check_only")
    
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        if e.response is not None:
            print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

