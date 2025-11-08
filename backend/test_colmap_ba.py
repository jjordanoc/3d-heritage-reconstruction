"""
Test script for COLMAP Bundle Adjustment Pipeline

Usage:
    python test_colmap_ba.py --scene_id auditorio --base_url http://your-app.modal.run
"""

import requests
import time
import argparse
import json


def trigger_colmap_ba(base_url: str, scene_id: str) -> dict:
    """Trigger COLMAP bundle adjustment processing."""
    url = f"{base_url}/scene/{scene_id}/colmap_ba"
    
    print(f"🚀 Triggering COLMAP BA for scene: {scene_id}")
    print(f"   URL: {url}")
    
    response = requests.post(url)
    response.raise_for_status()
    
    result = response.json()
    print(f"✅ Response: {json.dumps(result, indent=2)}")
    
    return result


def check_colmap_ba_status(base_url: str, scene_id: str) -> dict:
    """Check COLMAP bundle adjustment status."""
    url = f"{base_url}/scene/{scene_id}/colmap_ba/status"
    
    response = requests.get(url)
    response.raise_for_status()
    
    return response.json()


def wait_for_completion(base_url: str, scene_id: str, timeout: int = 600, poll_interval: int = 10):
    """
    Wait for COLMAP BA processing to complete.
    
    Args:
        base_url: Base URL of the API
        scene_id: Scene ID
        timeout: Maximum time to wait in seconds
        poll_interval: Time between status checks in seconds
    """
    print(f"\n⏳ Waiting for processing to complete (timeout: {timeout}s)...")
    
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed > timeout:
            print(f"❌ Timeout after {timeout}s")
            return None
        
        status_result = check_colmap_ba_status(base_url, scene_id)
        status = status_result.get('status')
        
        if status == 'complete':
            print(f"\n✅ Processing complete after {elapsed:.1f}s")
            print(f"\nResults:")
            print(json.dumps(status_result, indent=2))
            return status_result
        
        elif status == 'processing':
            print(f"   [{elapsed:.1f}s] Still processing... {status_result.get('message', '')}")
        
        elif status == 'not_started':
            print(f"   [{elapsed:.1f}s] Not started or failed: {status_result.get('message', '')}")
        
        else:
            print(f"   [{elapsed:.1f}s] Unknown status: {status}")
        
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Test COLMAP Bundle Adjustment Pipeline")
    parser.add_argument(
        "--scene_id",
        type=str,
        required=True,
        help="Scene ID to process (e.g., 'auditorio')"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        required=True,
        help="Base URL of the API (e.g., 'http://your-app.modal.run')"
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Only check status without triggering processing"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for processing to complete"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds when waiting (default: 600)"
    )
    
    args = parser.parse_args()
    
    # Remove trailing slash from base_url
    base_url = args.base_url.rstrip('/')
    
    try:
        if not args.check_only:
            # Trigger processing
            trigger_result = trigger_colmap_ba(base_url, args.scene_id)
            print(f"\n📋 Check status at: {base_url}{trigger_result.get('check_status_at', '')}")
        
        if args.wait or args.check_only:
            # Wait for completion or just check status
            if args.wait and not args.check_only:
                print("\nWaiting a few seconds for processing to start...")
                time.sleep(5)
            
            result = wait_for_completion(
                base_url,
                args.scene_id,
                timeout=args.timeout,
                poll_interval=10
            )
            
            if result and result.get('status') == 'complete':
                # Print summary
                print("\n" + "="*80)
                print("SUMMARY")
                print("="*80)
                
                ba_results = result.get('results', {})
                initial_error = ba_results.get('initial_error', {})
                final_error = ba_results.get('final_error', {})
                
                print(f"\nScene ID: {args.scene_id}")
                print(f"Status: {result.get('status')}")
                print(f"\nReconstructions:")
                print(f"  Initial: {result.get('initial_reconstruction')}")
                print(f"  Refined: {result.get('refined_reconstruction')}")
                
                print(f"\nReprojection Error:")
                print(f"  Initial: {initial_error.get('mean', 'N/A'):.3f} ± {initial_error.get('std', 'N/A'):.3f} px")
                print(f"  Final:   {final_error.get('mean', 'N/A'):.3f} ± {final_error.get('std', 'N/A'):.3f} px")
                
                print(f"\nReconstruction:")
                print(f"  Cameras: {ba_results.get('num_cameras', 'N/A')}")
                print(f"  Images:  {ba_results.get('num_images', 'N/A')}")
                print(f"  Points:  {ba_results.get('num_points3D', 'N/A')}")
                
                camera_params = ba_results.get('camera_params', {})
                if camera_params:
                    print(f"\nCamera Parameters:")
                    for cam_id, cam_data in camera_params.items():
                        print(f"  Camera {cam_id}: {cam_data['model']}")
                        print(f"    Size: {cam_data['width']}x{cam_data['height']}")
                        if cam_data['model'] == 'PINHOLE' and len(cam_data['params']) >= 4:
                            fx, fy, cx, cy = cam_data['params'][:4]
                            print(f"    fx={fx:.2f}, fy={fy:.2f}")
                            print(f"    cx={cx:.2f}, cy={cy:.2f}")
                
                print("\n" + "="*80)
        
        else:
            # Just triggered, not waiting
            print("\n✅ Processing started. Check status with:")
            print(f"   python {__file__} --scene_id {args.scene_id} --base_url {base_url} --check_only")
    
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

