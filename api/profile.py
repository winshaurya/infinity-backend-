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

        if method == 'GET':
            return get_user_profile(user_id)

        return {'error': 'Method not allowed'}, 405

    except Exception as e:
        return {'error': str(e)}, 500

def get_user_profile(user_id):
    response = supabase.table('users').select('*').eq('id', user_id).execute()
    if not response.data:
        return {'error': 'User not found'}, 404

    user = response.data[0]

    # Get recent activity
    activity_response = supabase.table('user_activity_history').select('*').eq('user_id', user_id).order('activity_date', desc=True).limit(10).execute()
    activities = activity_response.data or []

    tier = 'free'
    if user['is_fullaccess']:
        tier = 'fullaccess'
    elif user['credits'] >= 15:
        tier = 'paid'

    return {
        'user_id': user['id'],
        'email': user['email'],
        'credits': user['credits'],
        'subscription_tier': tier,
        'is_fullaccess': user['is_fullaccess'],
        'created_at': user['created_at'],
        'recent_activity': [{
            'activity_type': a['activity_type'],
            'details': a['details'],
            'credits_amount': a['credits_amount'],
            'activity_date': a['activity_date']
        } for a in activities]
    }