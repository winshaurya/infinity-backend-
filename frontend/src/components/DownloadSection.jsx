import { useState } from 'react'

function DownloadSection({ user, jobs, profile, onDownload, onRefillCredits }) {
  const [downloadForm, setDownloadForm] = useState({
    jobId: '',
    moleculesCount: 1000,
    format: 'csv'
  })
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadProgress, setDownloadProgress] = useState(0)

  const handleDownload = async () => {
    setIsDownloading(true)
    setDownloadProgress(0)

    // Simulate progress for SDF rendering
    if (downloadForm.format === 'sdf') {
      const progressInterval = setInterval(() => {
        setDownloadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + 10
        })
      }, 200)
    }

    try {
      await onDownload(downloadForm)
      setDownloadProgress(100)
    } finally {
      setTimeout(() => {
        setIsDownloading(false)
        setDownloadProgress(0)
      }, 1000)
    }
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
        <button onClick={handleDownload} disabled={!downloadForm.jobId || isDownloading}>
          {isDownloading ? 'Downloading...' : 'Download'}
        </button>
        {downloadForm.moleculesCount > 0 && (
          <span className="credit-cost">
            Cost: {Math.ceil(downloadForm.moleculesCount / 1000)} credits
          </span>
        )}
      </div>

      {isDownloading && downloadForm.format === 'sdf' && (
        <div className="download-progress">
          <div className="progress-text">
            Rendering compounds... {downloadProgress}%
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${downloadProgress}%` }}
            ></div>
          </div>
        </div>
      )}
    </div>
  )
}

export default DownloadSection