import json
import os
import sys
from supabase import create_client, Client
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from core_logic import generate_functionalized_isomers

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

        path = request.path
        method = request.method

        if method == 'GET':
            if '/jobs/' in path and '/preview' in path:
                # /jobs/{job_id}/preview
                job_id = path.split('/jobs/')[1].split('/preview')[0]
                return get_job_preview(job_id, user_id)
            elif '/jobs/' in path and '/summary' not in path:
                # /jobs/{job_id}
                job_id = path.split('/jobs/')[1]
                return get_job_status(job_id, user_id)
            elif '/jobs/summary' in path:
                # /jobs/summary
                return get_jobs_summary(user_id)
            else:
                # /jobs
                limit = int(request.args.get('limit', 20))
                offset = int(request.args.get('offset', 0))
                return get_user_jobs(user_id, limit, offset)

        return {'error': 'Method not allowed'}, 405

    except Exception as e:
        return {'error': str(e)}, 500

def get_job_status(job_id, user_id):
    response = supabase.table('jobs').select('*').eq('id', job_id).eq('user_id', user_id).execute()
    if not response.data:
        return {'error': 'Job not found'}, 404

    job = response.data[0]
    return {
        'job_id': job['id'],
        'status': job['status'],
        'total_molecules': job['total_molecules'],
        'created_at': job['created_at'],
        'completed_at': job['completed_at'],
        'parameters': json.loads(job['parameters']) if job['parameters'] else None
    }

def get_job_preview(job_id, user_id):
    job = get_job_status(job_id, user_id)
    if 'error' in job:
        return job

    if job['status'] != 'completed':
        return {'error': 'Job not completed'}, 400

    params = job['parameters']
    smiles_list = generate_functionalized_isomers(
        n_carbons=params["carbon_count"],
        functional_groups=params["functional_groups"],
        n_double_bonds=params["double_bonds"],
        n_triple_bonds=params["triple_bonds"],
        n_rings=params["rings"],
        carbon_types=params["carbon_types"]
    )

    preview = smiles_list[:3] if len(smiles_list) >= 3 else smiles_list
    return {
        'job_id': job_id,
        'status': 'completed',
        'preview_molecules': preview,
        'preview_count': len(preview),
        'total_molecules': job['total_molecules']
    }

def get_user_jobs(user_id, limit, offset):
    response = supabase.table('jobs').select('*').eq('user_id', user_id).order('created_at', desc=True).range(offset, offset + limit - 1).execute()
    jobs = response.data or []

    count_response = supabase.table('jobs').select('id', count='exact').eq('user_id', user_id).execute()
    total_count = count_response.count

    return {
        'jobs': [{
            'id': job['id'],
            'parameters': json.loads(job['parameters']),
            'status': job['status'],
            'total_molecules': job['total_molecules'],
            'created_at': job['created_at'],
            'updated_at': job['updated_at'],
            'completed_at': job['completed_at']
        } for job in jobs],
        'total_count': total_count,
        'count': len(jobs),
        'limit': limit,
        'offset': offset,
        'has_more': (offset + limit) < total_count
    }

def get_jobs_summary(user_id):
    response = supabase.table('jobs').select('status').eq('user_id', user_id).execute()
    jobs = response.data or []

    status_counts = {}
    for job in jobs:
        status = job['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    for status in ['pending', 'processing', 'completed', 'failed']:
        if status not in status_counts:
            status_counts[status] = 0

    active_response = supabase.table('jobs').select('id,status,created_at').eq('user_id', user_id).in_('status', ['pending', 'processing']).order('created_at', desc=True).limit(5).execute()
    active_jobs = active_response.data or []

    return {
        'status_counts': status_counts,
        'active_jobs': active_jobs,
        'total_recent_jobs': len(jobs)
    }