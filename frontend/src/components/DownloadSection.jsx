import { useState } from 'react'

function DownloadSection({ user, jobs, profile, onDownload, onRefillCredits }) {
  const [downloadForm, setDownloadForm] = useState({
    jobId: '',
    moleculesCount: 1000,
    format: 'csv'
  })

  const handleDownload = () => {
    onDownload(downloadForm)
  }

  if (!user) return null

  return (
    <div className="download-section">
      <h3>Download Molecules</h3>
      <div className="download-form">
        <select
          value={downloadForm.jobId}
          onChange={(e) => setDownloadForm(prev => ({ ...prev, jobId: e.target.value }))}
        >
          <option value="">Select a completed job</option>
          {jobs.filter(job => job.status === 'completed').map(job => (
            <option key={job.id} value={job.id}>
              Job {job.id.slice(0, 8)} - {job.total_molecules} molecules ({new Date(job.created_at).toLocaleDateString()})
            </option>
          ))}
        </select>
        <input
          type="number"
          placeholder="Molecules to download"
          value={downloadForm.moleculesCount}
          onChange={(e) => setDownloadForm(prev => ({ ...prev, moleculesCount: parseInt(e.target.value) || 1000 }))}
          min="1"
          max="100000"
        />
        <select
          value={downloadForm.format}
          onChange={(e) => setDownloadForm(prev => ({ ...prev, format: e.target.value }))}
        >
          <option value="csv">CSV</option>
          <option value="sdf">SDF</option>
          {profile?.subscription_tier === 'fullaccess' && <option value="all">All (Fullaccess only)</option>}
        </select>
        <button onClick={handleDownload} disabled={!downloadForm.jobId}>
          Download
        </button>
        {downloadForm.moleculesCount > 0 && (
          <span className="credit-cost">
            Cost: {Math.ceil(downloadForm.moleculesCount / 1000)} credits
          </span>
        )}
      </div>
    </div>
  )
}

export default DownloadSection