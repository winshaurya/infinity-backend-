function JobStatus({ currentJob, jobs }) {
  return (
    <>
      {currentJob && (
        <div className="current-job">
          <h3>Current Job</h3>
          <div className="job-info">
            <p><strong>Job ID:</strong> {currentJob.id}</p>
            <p><strong>Status:</strong> {currentJob.status}</p>
            <p><strong>Molecules:</strong> {currentJob.total_molecules || 0}</p>
            {currentJob.parameters && (
              <p><strong>Parameters:</strong> {JSON.stringify(currentJob.parameters, null, 2)}</p>
            )}
          </div>
        </div>
      )}

      {jobs.length > 0 && (
        <div className="jobs-history">
          <h3>Generation History</h3>
          <div className="jobs-list">
            {jobs.map(job => (
              <div key={job.id} className={`job-item ${job.status}`}>
                <div className="job-header">
                  <span className="job-id">Job {job.id.slice(0, 8)}</span>
                  <span className={`job-status ${job.status}`}>{job.status}</span>
                </div>
                <div className="job-details">
                  <span>Molecules: {job.total_molecules}</span>
                  <span>Created: {new Date(job.created_at).toLocaleString()}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  )
}

export default JobStatus