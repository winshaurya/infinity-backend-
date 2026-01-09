# Chemistry SaaS Performance Test

This test script benchmarks the molecule generation performance, specifically testing 16-carbon molecule generation.

## Prerequisites

1. **Backend must be running** on `http://localhost:8000`
2. **Database must be set up** with proper schema
3. **User authentication** must be working
4. **Test user account** with credits

## Setup

1. **Get User Token:**
   - Start the frontend application
   - Login with a test user account
   - Open Browser Developer Tools (F12)
   - Go to Network tab
   - Make any authenticated request
   - Copy the `Authorization` header value (starts with "Bearer ")

2. **Set Environment Variable:**
   ```bash
   export TEST_USER_TOKEN="your_token_here"
   ```

   Or create a `.env` file in the project root:
   ```
   TEST_USER_TOKEN=your_token_here
   ```

## Running the Test

```bash
python test_generation_performance.py
```

## What the Test Does

1. **Generates 16-carbon molecules** with:
   - 16 carbons
   - 2 double bonds
   - 1 ring
   - 2 OH functional groups

2. **Measures performance:**
   - Total generation time
   - Number of molecules generated
   - Molecules per second

3. **Tests download system:**
   - Calculates credit cost for downloading 1000 molecules
   - Verifies credit deduction works

## Expected Output

```
🧪 Chemistry SaaS Generation Performance Test
==================================================
🕐 Started at: 2026-01-09 15:18:00
🌐 API Base: http://localhost:8000
🚀 Starting 16-carbon molecule generation test...
📊 Parameters: {
  "carbon_count": 16,
  "double_bonds": 2,
  "triple_bonds": 0,
  "rings": 1,
  "carbon_types": ["primary", "secondary", "tertiary"],
  "functional_groups": ["OH", "OH"]
}
--------------------------------------------------
📤 Sending generation request...
✅ Generation started! Job ID: abc123...
⏳ Polling for completion...
📊 Status: completed | Molecules: 15432
✅ Generation completed!
⏱️  Total time: 45.67 seconds
🧪 Molecules generated: 15432
⚡ Performance: 338.12 molecules/second

📥 Testing download cost calculation...
💰 Download cost: 16 credits
💳 Remaining credits: 84

🏁 Test completed in 45.67 seconds
==================================================
🏁 All tests completed at: 2026-01-09 15:18:45
```

## Troubleshooting

- **"Please set TEST_USER_TOKEN environment variable"**: Follow setup step 2
- **Connection refused**: Make sure backend is running on port 8000
- **Authentication failed**: Check token is valid and not expired
- **Generation failed**: Check backend logs for errors
- **No credits**: Add credits to test user account

## Test Parameters

The test uses realistic parameters for 16-carbon molecules:
- **Carbon count**: 16 (large molecule test)
- **Double bonds**: 2 (unsaturated)
- **Rings**: 1 (cyclic structure)
- **Functional groups**: 2 OH groups (diols)

This represents a challenging but realistic molecule generation task.