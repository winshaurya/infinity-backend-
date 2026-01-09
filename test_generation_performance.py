#!/usr/bin/env python3
"""
Test script to benchmark molecule generation performance
Tests generation of 16-carbon molecules and measures time
"""

import time
import requests
import json
import os
from datetime import datetime

# Configuration
API_BASE = 'http://localhost:8000'
TEST_USER_TOKEN = os.getenv('TEST_USER_TOKEN')  # You'll need to set this

def test_16_carbon_generation():
    """Test generation of 16-carbon molecules and measure performance"""

    # Test parameters for 16-carbon molecules
    test_payload = {
        "carbon_count": 16,
        "double_bonds": 2,
        "triple_bonds": 0,
        "rings": 1,
        "carbon_types": ["primary", "secondary", "tertiary"],
        "functional_groups": ["OH", "OH"]  # Two alcohol groups
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TEST_USER_TOKEN}'
    }

    print("🚀 Starting 16-carbon molecule generation test...")
    print(f"📊 Parameters: {json.dumps(test_payload, indent=2)}")
    print("-" * 50)

    # Start timing
    start_time = time.time()

    try:
        # Step 1: Start generation
        print("📤 Sending generation request...")
        response = requests.post(f"{API_BASE}/generate", json=test_payload, headers=headers)

        if response.status_code != 200:
            print(f"❌ Generation request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return

        data = response.json()
        job_id = data['job_id']
        print(f"✅ Generation started! Job ID: {job_id}")

        # Step 2: Poll for completion
        print("⏳ Polling for completion...")
        while True:
            status_response = requests.get(f"{API_BASE}/jobs/{job_id}", headers=headers)

            if status_response.status_code != 200:
                print(f"❌ Status check failed: {status_response.status_code}")
                break

            status_data = status_response.json()
            current_status = status_data['status']
            molecules_count = status_data.get('total_molecules', 0)

            print(f"📊 Status: {current_status} | Molecules: {molecules_count}", end='\r')

            if current_status == 'completed':
                generation_time = time.time() - start_time
                print(f"\n✅ Generation completed!")
                print(f"⏱️  Total time: {generation_time:.2f} seconds")
                print(f"🧪 Molecules generated: {molecules_count}")
                print(f"⚡ Performance: {molecules / generation_time:.2f} molecules/second")

                # Step 3: Test download (but don't actually download)
                print("\n📥 Testing download cost calculation...")
                download_payload = {
                    "job_id": job_id,
                    "molecules_count": min(1000, molecules_count),  # Test with 1000 molecules
                    "download_format": "csv"
                }

                download_response = requests.post(f"{API_BASE}/download",
                                                json=download_payload,
                                                headers=headers)

                if download_response.status_code == 200:
                    download_data = download_response.json()
                    print(f"💰 Download cost: {download_data.get('credits_used', 'N/A')} credits")
                    print(f"💳 Remaining credits: {download_data.get('remaining_credits', 'N/A')}")
                else:
                    print(f"⚠️  Download test failed: {download_response.status_code}")

                break

            elif current_status == 'failed':
                print(f"\n❌ Generation failed!")
                break

            time.sleep(2)  # Poll every 2 seconds

    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")

    total_time = time.time() - start_time
    print(f"\n🏁 Test completed in {total_time:.2f} seconds")
    print("=" * 50)

def run_multiple_tests():
    """Run multiple generation tests with different parameters"""
    test_cases = [
        {"carbon_count": 8, "double_bonds": 1, "triple_bonds": 0, "rings": 0, "functional_groups": ["OH"]},
        {"carbon_count": 12, "double_bonds": 2, "triple_bonds": 0, "rings": 1, "functional_groups": ["OH", "COOH"]},
        {"carbon_count": 16, "double_bonds": 2, "triple_bonds": 0, "rings": 1, "functional_groups": ["OH", "OH"]},
        {"carbon_count": 20, "double_bonds": 3, "triple_bonds": 0, "rings": 2, "functional_groups": ["OH", "COOH", "NH2"]},
    ]

    print("🧪 Running comprehensive generation performance tests...")
    print("=" * 60)

    for i, params in enumerate(test_cases, 1):
        print(f"\n🔬 Test {i}/{len(test_cases)}: {params['carbon_count']} carbons")
        test_single_generation(params)

def test_single_generation(params):
    """Test a single generation with given parameters"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TEST_USER_TOKEN}'
    }

    payload = {
        **params,
        "carbon_types": ["primary", "secondary", "tertiary"]
    }

    start_time = time.time()

    try:
        response = requests.post(f"{API_BASE}/generate", json=payload, headers=headers)

        if response.status_code != 200:
            print(f"❌ Failed: {response.status_code}")
            return

        data = response.json()
        job_id = data['job_id']

        # Poll for completion
        while True:
            status_response = requests.get(f"{API_BASE}/jobs/{job_id}", headers=headers)
            if status_response.status_code != 200:
                break

            status_data = status_response.json()
            if status_data['status'] == 'completed':
                generation_time = time.time() - start_time
                molecules = status_data.get('total_molecules', 0)
                print(f"✅ {params['carbon_count']}C: {molecules} molecules in {generation_time:.2f}s ({molecules / generation_time:.2f} mol/s)")
                break
            elif status_data['status'] == 'failed':
                print(f"❌ Failed after {time.time() - start_time:.2f}s")
                break

            time.sleep(1)

    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🧪 Chemistry SaaS Generation Performance Test")
    print("=" * 50)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API Base: {API_BASE}")

    if not TEST_USER_TOKEN:
        print("❌ Please set TEST_USER_TOKEN environment variable")
        print("💡 Get token from: Login to app -> Browser Dev Tools -> Network -> Authorization header")
        exit(1)

    # Run the main 16-carbon test
    test_16_carbon_generation()

    # Optional: Run comprehensive tests
    # print("\n🔬 Running comprehensive tests...")
    # run_multiple_tests()

    print(f"\n🏁 All tests completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")