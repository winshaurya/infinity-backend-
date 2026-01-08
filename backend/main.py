from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
import time
from datetime import datetime
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from core_logic import generate_functionalized_isomers, validate_structure_possibility, get_functional_group_type, get_element_counts, rdMolDescriptors
from rdkit import Chem

load_dotenv()

app = FastAPI(title="Chemistry SaaS API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase (optional for local dev)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Models
class JobRequest(BaseModel):
    carbon_count: int = Field(ge=0, le=20)
    double_bonds: int = Field(ge=0, le=10)
    triple_bonds: int = Field(ge=0, le=10)
    rings: int = Field(ge=0, le=10)
    carbon_types: List[str] = Field(default=["primary", "secondary", "tertiary"])
    functional_groups: List[str] = Field(max_items=50)

class UnlockRequest(BaseModel):
    job_id: str

# In-memory job storage (in production, use Redis or similar)
jobs = {}

# In-memory storage for molecules
job_results = {}

def generate_molecules(job_id: str, params: dict):
    try:
        # Update status to processing
        jobs[job_id]["status"] = "processing"

        # Generate SMILES
        smiles_list = generate_functionalized_isomers(
            n_carbons=params["carbon_count"],
            functional_groups=params["functional_groups"],
            n_double_bonds=params["double_bonds"],
            n_triple_bonds=params["triple_bonds"],
            n_rings=params["rings"],
            carbon_types=params["carbon_types"]
        )

        # Store in memory
        job_results[job_id] = smiles_list

        # Update job
        total_molecules = len(smiles_list)
        credits_needed = (total_molecules // 1000) + 1  # 1 credit per 1000 molecules
        jobs[job_id].update({
            "status": "completed",
            "total_molecules": total_molecules,
            "credits_needed": credits_needed
        })

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        print(f"Error generating molecules: {e}")

@app.post("/generate")
async def start_generation(request: JobRequest, background_tasks: BackgroundTasks):
    # Validate
    is_valid, error = validate_structure_possibility(
        request.carbon_count, request.functional_groups,
        request.double_bonds, request.triple_bonds, request.carbon_types, request.rings
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    # Create job
    job_id = str(uuid.uuid4())
    job_data = {
        "id": job_id,
        "user_id": "test_user",  # In production, get from auth
        "parameters": request.dict(),
        "status": "pending",
        "total_molecules": 0,
        "created_at": datetime.utcnow().isoformat()
    }
    jobs[job_id] = job_data

    # Start background task
    background_tasks.add_task(generate_molecules, job_id, request.dict())

    return {"job_id": job_id, "status": "started"}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {
        "status": job["status"],
        "total_molecules": job.get("total_molecules", 0),
        "progress": 100 if job["status"] == "completed" else 50  # Simplified
    }

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    # Check if job exists and completed
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] not in ["completed", "unlocked"]:
        raise HTTPException(status_code=400, detail="Job not completed")

    # Get from memory
    smiles_list = job_results.get(job_id, [])
    if job["status"] == "unlocked":
        return {"status": "unlocked", "smiles": smiles_list}
    else:
        # Preview: first 3
        return {"status": "completed", "smiles": smiles_list[:3]}

@app.post("/unlock/{job_id}")
async def unlock_job(job_id: str):
    # Check job
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    # Check credits (simplified)
    credits_needed = job.get("credits_needed", 1)
    # In production, check user's credits and deduct
    # For now, assume sufficient

    # Update to unlocked
    job["status"] = "unlocked"

    return {"status": "unlocked"}

@app.get("/profile")
async def get_profile():
    # Simplified profile
    return {"credits": 100, "email": "test@example.com"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
