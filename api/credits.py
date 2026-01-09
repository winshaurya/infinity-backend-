import json
import os
import sys
from supabase import create_client, Client
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_user_id(request):
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    token = auth_header.split(' ')[1]
    try:
        user_response = supabase.auth.get_user(token)
        return user_response.user.id
    except:
        return None

def handler(request):
    try:
        user_id = get_user_id(request)
        if not user_id:
            return {'error': 'Unauthorized'}, 401

        method = request.method

        if method == 'POST':
            data = request.get_json()
            amount = data.get('amount', 0)
            description = data.get('description', 'Manual credit refill')

            return refill_credits(user_id, amount, description)

        return {'error': 'Method not allowed'}, 405

    except Exception as e:
        return {'error': str(e)}, 500

def refill_credits(user_id, amount, description):
    if amount <= 0:
        return {'error': 'Amount must be positive'}, 400

    # Get current credits
    response = supabase.table('users').select('credits').eq('id', user_id).execute()
    if not response.data:
        return {'error': 'User not found'}, 404

    current_credits = response.data[0]['credits']
    new_credits = current_credits + amount

    # Update credits
    supabase.table('users').update({'credits': new_credits}).eq('id', user_id).execute()

    # Add credit history
    supabase.table('credit_history').insert({
        'user_id': user_id,
        'amount': amount,
        'reason': 'Manual refill',
        'description': description
    }).execute()

    return {
        'success': True,
        'credits_added': amount,
        'new_balance': new_credits,
        'message': f'Added {amount} credits to account'
    }