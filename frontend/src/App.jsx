import { useState, useEffect } from 'react'
import { createClient } from '@supabase/supabase-js'
import { RDKit } from '@rdkit/rdkit'
import './App.css'

const API_BASE = 'http://localhost:8000' // Change to your backend URL

let supabase
try {
  supabase = createClient(import.meta.env.VITE_SUPABASE_URL || 'dummy', import.meta.env.VITE_SUPABASE_ANON_KEY || 'dummy')
} catch (error) {
  console.error('Supabase init failed:', error)
  supabase = null
}

function App() {
  console.log('App component rendering')
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [showAuthModal, setShowAuthModal] = useState(false)
  const [loginData, setLoginData] = useState({ email: '', password: '' })
  const [isSignUp, setIsSignUp] = useState(false)

  const [rdkit, setRdkit] = useState(null)
  const [formData, setFormData] = useState({
    carbon_count: 0,
    double_bonds: 0,
    triple_bonds: 0,
    rings: 0,
    carbon_types: ['primary', 'secondary', 'tertiary'],
    functional_groups: {}
  })
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState('')
  const [progress, setProgress] = useState(0)
  const [molecules, setMolecules] = useState([])
  const [allSmiles, setAllSmiles] = useState([])
  const [profile, setProfile] = useState({ credits: 100, email: 'test@example.com' })
  const [showProfile, setShowProfile] = useState(false)
  const [profileTab, setProfileTab] = useState('profile')
  const [passwordData, setPasswordData] = useState({ current: '', new: '', confirm: '' })
  const [disabledGroups, setDisabledGroups] = useState({})
  const [valencyStatus, setValencyStatus] = useState('Valency status: OK')
  const [toast, setToast] = useState(null)
  const [downloadCount, setDownloadCount] = useState(1) // in thousands
  const [generating, setGenerating] = useState(false)
  const [unlocking, setUnlocking] = useState(false)
  const [downloading, setDownloading] = useState(false)

  const functionalGroups = [
    'Amide', 'Azide', 'Br', 'CHO', 'Cl', 'CN', 'COOR_C2H5', 'COOR_C3H7', 'COOR_CH(CH3)2', 'COOR_CH3',
    'COOH', 'COX_Br', 'COX_Cl', 'COX_F', 'COX_I', 'Ether', 'F', 'I', 'Imine', 'Ketone',
    'NC', 'NCO', 'NH2', 'NO2', 'OCN', 'OH', 'OX_Br', 'OX_Cl', 'OX_F', 'OX_I',
    'S_Bivalent', 'S_Tetravalent', 'S_Hexavalent', 'S_Chain_Bi', 'S_Chain_Tetra', 'S_Chain_Hexa'
  ]

  useEffect(() => {
    const checkUser = async () => {
      if (supabase) {
        const { data: { user } } = await supabase.auth.getUser()
        setUser(user)
      }
      setLoading(false)
    }
    checkUser()

    let subscription = null
    if (supabase) {
      const { data: { subscription: sub } } = supabase.auth.onAuthStateChange((_event, session) => {
        setUser(session?.user ?? null)
        setLoading(false)
      })
      subscription = sub
    }

    return () => subscription?.unsubscribe()
  }, [])

  // useEffect(() => {
  //   const initRDKit = async () => {
  //     try {
  //       const rdkitInstance = await RDKit.load()
  //       setRdkit(rdkitInstance)
  //     } catch (error) {
  //       console.error('Failed to load RDKit:', error)
  //       // Continue without RDKit
  //     }
  //   }
  //   initRDKit()
  // }, [])

  useEffect(() => {
    updateFunctionalGroupStates()
    updateValencyStatus()
  }, [formData.carbon_count, formData.double_bonds, formData.triple_bonds, formData.rings, formData.functional_groups])

  const updateFunctionalGroupStates = () => {
    const carbonCount = formData.carbon_count
    const disabled = {}

    const disabledForZeroCarbon = new Set([
      'Amide', 'Cl', 'Br', 'I', 'F', 'CN', 'NC', 'OCN', 'NCO', 'Imine', 'NO2', 'Ketone', 'Ether', 'OH', 'NH2', 'OX_Cl', 'OX_Br', 'OX_F', 'OX_I', 'Azide',
      'S_Bivalent', 'S_Tetravalent', 'S_Hexavalent', 'S_Chain_Bi', 'S_Chain_Tetra', 'S_Chain_Hexa'
    ])

    const alwaysAllowed = new Set([
      'COOH', 'CHO', 'COOR_CH3', 'COOR_C2H5', 'COOR_C3H7', 'COOR_CH(CH3)2', 'COX_Cl', 'COX_Br', 'COX_F', 'COX_I'
    ])

    if (carbonCount === 0) {
      functionalGroups.forEach(fg => {
        if (disabledForZeroCarbon.has(fg) && !alwaysAllowed.has(fg)) {
          disabled[fg] = true
        }
      })
    } else {
      if (carbonCount < 2) {
        disabled['Ether'] = true
        disabled['Ketone'] = true
        disabled['S_Chain_Bi'] = true
        disabled['S_Chain_Tetra'] = true
        disabled['S_Chain_Hexa'] = true
      }
      if (carbonCount === 1) {
        disabled['Ketone'] = true
      }
    }

    setDisabledGroups(disabled)
  }

  const updateValencyStatus = () => {
    const carbonCount = formData.carbon_count
    const nDoubleBonds = formData.double_bonds
    const nTripleBonds = formData.triple_bonds
    const nRings = formData.rings

    if (carbonCount === 0) {
      setValencyStatus('Valency status: OK')
      return
    }

    let maxValency = 2 * carbonCount + 2 - 2 * nDoubleBonds - 4 * nTripleBonds
    if (nRings > 0) {
      maxValency += nRings
    }

    let currentValency = 0
    Object.values(formData.functional_groups).forEach(count => {
      currentValency += count
    })

    if (currentValency > maxValency) {
      setValencyStatus(`Valency status: EXCEEDED (Current: ${currentValency}, Max: ${maxValency})`)
    } else if (currentValency === maxValency) {
      setValencyStatus(`Valency status: MAXIMUM REACHED (Current: ${currentValency}, Max: ${maxValency})`)
    } else {
      setValencyStatus(`Valency status: OK (Current: ${currentValency}, Max: ${maxValency})`)
    }
  }

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  const handleFunctionalGroupChange = (fg, value) => {
    const count = parseInt(value) || 0
    const carbonCount = formData.carbon_count
    const nDoubleBonds = formData.double_bonds
    const nTripleBonds = formData.triple_bonds
    const nRings = formData.rings

    let maxValency = 2 * carbonCount + 2 - 2 * nDoubleBonds - 4 * nTripleBonds
    if (nRings > 0) {
      maxValency += nRings
    }

    let currentValency = 0
    Object.entries(formData.functional_groups).forEach(([key, val]) => {
      if (key !== fg) currentValency += val
    })
    currentValency += count

    if (currentValency > maxValency) {
      alert(`Cannot add more functional groups. The maximum allowed is ${maxValency}.`)
      return
    }

    setFormData(prev => ({
      ...prev,
      functional_groups: { ...prev.functional_groups, [fg]: count }
    }))
  }

  const handleCarbonTypeChange = (type, checked) => {
    setFormData(prev => ({
      ...prev,
      carbon_types: checked
        ? [...prev.carbon_types, type]
        : prev.carbon_types.filter(t => t !== type)
    }))
  }

  const generateMolecules = async () => {
    if (!user) {
      setShowAuthModal(true)
      return
    }

    setGenerating(true)
    const payload = {
      carbon_count: formData.carbon_count,
      double_bonds: formData.double_bonds,
      triple_bonds: formData.triple_bonds,
      rings: formData.rings,
      carbon_types: formData.carbon_types,
      functional_groups: Object.entries(formData.functional_groups)
        .filter(([_, count]) => count > 0)
        .flatMap(([fg, count]) => Array(count).fill(fg))
    }

    try {
      const response = await fetch(`${API_BASE}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      const data = await response.json()
      setJobId(data.job_id)
      setStatus('Generating...')
      pollStatus(data.job_id)
    } catch (error) {
      setStatus('Error: ' + error.message)
      showToast('Failed to start generation: ' + error.message)
    } finally {
      setGenerating(false)
    }
  }

  const pollStatus = async (id) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/status/${id}`)
        const data = await response.json()
        setStatus(`Status: ${data.status}, Molecules: ${data.total_molecules}`)
        setProgress(data.progress)
        if (data.status === 'completed') {
          clearInterval(interval)
          // Optionally unlock or show preview
        }
      } catch (error) {
        console.error('Polling error:', error)
      }
    }, 2000)
  }

  const unlockJob = async () => {
    if (!jobId) return
    setUnlocking(true)
    try {
      await fetch(`${API_BASE}/unlock/${jobId}`, { method: 'POST' })
      // Fetch full results
      const response = await fetch(`${API_BASE}/results/${jobId}`)
      const data = await response.json()
      setAllSmiles(data.smiles)
      setMolecules(data.smiles.slice(0, 3)) // Still show first 3, but now all available for download
    } catch (error) {
      console.error('Unlock error:', error)
      showToast('Failed to unlock results: ' + error.message)
    } finally {
      setUnlocking(false)
    }
  }

  const downloadSDF = () => {
    if (!rdkit || !allSmiles.length) return
    setDownloading(true)
    try {
      let sdf = ''
      const toDownload = allSmiles.slice(0, downloadCount * 1000)
      toDownload.forEach((smiles, idx) => {
        const mol = rdkit.get_mol(smiles)
        if (mol) {
          sdf += mol.get_molblock() + '\n$$$$\n'
        }
      })
      const blob = new Blob([sdf], { type: 'chemical/x-mdl-sdfile' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `molecules_${downloadCount}k.sdf`
      a.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      showToast('Download failed: ' + error.message)
    } finally {
      setDownloading(false)
    }
  }

  const downloadCSV = () => {
    if (!allSmiles.length) return
    setDownloading(true)
    try {
      let csv = 'SMILES\n'
      const toDownload = allSmiles.slice(0, downloadCount * 1000)
      toDownload.forEach(smiles => {
        csv += `${smiles}\n`
      })
      const blob = new Blob([csv], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `molecules_${downloadCount}k.csv`
      a.click()
      URL.revokeObjectURL(url)
    } catch (error) {
      showToast('Download failed: ' + error.message)
    } finally {
      setDownloading(false)
    }
  }

  const handleLogin = async (e) => {
    e.preventDefault()
    const { error } = await supabase.auth.signInWithPassword({
      email: loginData.email,
      password: loginData.password
    })
    if (error) {
      showToast('Login failed: ' + error.message)
    } else {
      setShowAuthModal(false)
      setLoginData({ email: '', password: '' })
    }
  }

  const handleSignUp = async (e) => {
    e.preventDefault()
    const { error } = await supabase.auth.signUp({
      email: loginData.email,
      password: loginData.password
    })
    if (error) {
      showToast('Sign up failed: ' + error.message)
    } else {
      showToast('Check your email for confirmation', 'success')
      setShowAuthModal(false)
      setLoginData({ email: '', password: '' })
    }
  }

  const handleLogout = async () => {
    if (supabase) {
      await supabase.auth.signOut()
    }
  }

  const handleForgotPassword = async () => {
    if (!loginData.email) {
      showToast('Please enter your email first')
      return
    }
    const { error } = await supabase.auth.resetPasswordForEmail(loginData.email, {
      redirectTo: `${window.location.origin}/reset-password`
    })
    if (error) {
      showToast('Password reset failed: ' + error.message)
    } else {
      showToast('Password reset email sent! Check your inbox.', 'success')
    }
  }

  const handleChangePassword = async (e) => {
    e.preventDefault()
    if (passwordData.new !== passwordData.confirm) {
      showToast('New passwords do not match')
      return
    }
    const { error } = await supabase.auth.updateUser({
      password: passwordData.new
    })
    if (error) {
      showToast('Password change failed: ' + error.message)
    } else {
      showToast('Password changed successfully!', 'success')
      setPasswordData({ current: '', new: '', confirm: '' })
    }
  }

  const showToast = (message, type = 'error') => {
    setToast({ message, type })
    setTimeout(() => setToast(null), 5000)
  }

  if (loading) return <div className="app"><div className="container">Loading...</div></div>

  return (
    <div className="app">
      <div className="profile-button" onClick={() => user ? setShowProfile(!showProfile) : setShowAuthModal(true)}>
        {user ? '👤' : '🔐'}
      </div>

      {showProfile && (
        <div className="modal-overlay" onClick={() => setShowProfile(false)}>
          <div className="profile-modal" onClick={(e) => e.stopPropagation()}>
            <div className="profile-header">
              <h2>Account Settings</h2>
              <button onClick={() => setShowProfile(false)} className="close-btn">×</button>
            </div>
            <div className="profile-tabs">
              <button
                className={profileTab === 'profile' ? 'active' : ''}
                onClick={() => setProfileTab('profile')}
              >
                Profile
              </button>
              <button
                className={profileTab === 'password' ? 'active' : ''}
                onClick={() => setProfileTab('password')}
              >
                Password
              </button>
              <button
                className={profileTab === 'payment' ? 'active' : ''}
                onClick={() => setProfileTab('payment')}
              >
                Payment
              </button>
            </div>
            <div className="profile-content">
              {profileTab === 'profile' && (
                <div className="profile-section">
                  <h3>Profile Information</h3>
                  <div className="profile-field">
                    <label>Email:</label>
                    <span>{user?.email || 'Not logged in'}</span>
                  </div>
                  <div className="profile-field">
                    <label>Credits:</label>
                    <span>{profile.credits}</span>
                  </div>
                  <div className="profile-field">
                    <label>Account Status:</label>
                    <span>Active</span>
                  </div>
                  <button onClick={handleLogout} className="logout-btn">Logout</button>
                </div>
              )}
              {profileTab === 'password' && (
                <div className="profile-section">
                  <h3>Change Password</h3>
                  <form onSubmit={handleChangePassword}>
                    <div className="form-group">
                      <label>New Password:</label>
                      <input
                        type="password"
                        value={passwordData.new}
                        onChange={(e) => setPasswordData(prev => ({ ...prev, new: e.target.value }))}
                        required
                      />
                    </div>
                    <div className="form-group">
                      <label>Confirm New Password:</label>
                      <input
                        type="password"
                        value={passwordData.confirm}
                        onChange={(e) => setPasswordData(prev => ({ ...prev, confirm: e.target.value }))}
                        required
                      />
                    </div>
                    <button type="submit">Update Password</button>
                  </form>
                </div>
              )}
              {profileTab === 'payment' && (
                <div className="profile-section">
                  <h3>Payment & Billing</h3>
                  <div className="profile-field">
                    <label>Current Plan:</label>
                    <span>Free (100 credits)</span>
                  </div>
                  <div className="profile-field">
                    <label>Credits Used:</label>
                    <span>0</span>
                  </div>
                  <div className="profile-field">
                    <label>Credits Remaining:</label>
                    <span>{profile.credits}</span>
                  </div>
                  <button className="upgrade-btn">Upgrade Plan</button>
                  <button className="billing-btn">View Billing History</button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="container">
        <div className="header">
          <h1>CHEM-∞ : The Universal Molecular Generator</h1>
        </div>
        <div className="system-info">System: 4 CPU cores, 16GB RAM</div>

        <div className="form-grid">
          <label>Number of Carbons (0-20):</label>
          <input
            type="number"
            min="0"
            max="20"
            value={formData.carbon_count}
            onChange={(e) => handleInputChange('carbon_count', parseInt(e.target.value) || 0)}
          />

          <label>Number of Double Bonds:</label>
          <input
            type="number"
            min="0"
            max="10"
            value={formData.double_bonds}
            onChange={(e) => handleInputChange('double_bonds', parseInt(e.target.value) || 0)}
          />

          <label>Number of Triple Bonds:</label>
          <input
            type="number"
            min="0"
            max="10"
            value={formData.triple_bonds}
            onChange={(e) => handleInputChange('triple_bonds', parseInt(e.target.value) || 0)}
          />

          <label>Number of Rings:</label>
          <input
            type="number"
            min="0"
            max="10"
            value={formData.rings}
            onChange={(e) => handleInputChange('rings', parseInt(e.target.value) || 0)}
          />

          <label>Carbon Types:</label>
          <div className="carbon-types">
            {['primary', 'secondary', 'tertiary'].map(type => (
              <label key={type}>
                <input
                  type="checkbox"
                  checked={formData.carbon_types.includes(type)}
                  onChange={(e) => handleCarbonTypeChange(type, e.target.checked)}
                />
                {type}
              </label>
            ))}
          </div>
        </div>

        <div className="functional-groups-title">Functional Groups (Counts):</div>
        <div className="functional-groups">
          {functionalGroups.map(fg => (
            <div key={fg} className="fg-item">
              <label>{fg}:</label>
              <input
                type="number"
                min="0"
                max="10"
                value={formData.functional_groups[fg] || 0}
                onChange={(e) => handleFunctionalGroupChange(fg, e.target.value)}
                disabled={disabledGroups[fg]}
              />
            </div>
          ))}
        </div>

        <div className="valency-status">{valencyStatus}</div>

        <label>CPU Cores to use:</label>
        <div className="cpu-cores">
          <label><input type="radio" name="cpu" value="4" defaultChecked /> All 4 cores</label>
          <label><input type="radio" name="cpu" value="2" /> Half (2)</label>
          <label><input type="radio" name="cpu" value="custom" /> Custom:</label>
          <input type="number" min="1" max="4" defaultValue="4" style={{ width: '50px' }} />
        </div>

        <label>Select output directory:</label>
        <div className="output-dir">
          <input type="text" placeholder="Select output directory" />
          <button>Browse...</button>
        </div>

        <div className="buttons">
          <button>Clear Cache</button>
          <button onClick={generateMolecules} disabled={generating}>
            {generating ? 'Generating...' : 'GENERATE COMPOUND STRUCTURES'}
          </button>
          <button onClick={() => setShowProfile(true)}>Profile</button>
          {jobId && <button onClick={unlockJob} disabled={unlocking}>
            {unlocking ? 'Unlocking...' : 'Unlock Results'}
          </button>}
          {allSmiles.length > 0 && (
            <>
              <div>
                <label>Download Count (in 1000s):</label>
                <input
                  type="number"
                  min="1"
                  max={Math.ceil(allSmiles.length / 1000)}
                  value={downloadCount}
                  onChange={(e) => setDownloadCount(parseInt(e.target.value) || 1)}
                />
                <span>Credits needed: {downloadCount}</span>
              </div>
              <button onClick={downloadSDF} disabled={downloading}>
                {downloading ? 'Downloading...' : 'Download SDF'}
              </button>
              <button onClick={downloadCSV} disabled={downloading}>
                {downloading ? 'Downloading...' : 'Download CSV'}
              </button>
            </>
          )}
        </div>

        <div className="status">{status}</div>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }}></div>
        </div>

        <div className="perf-label"></div>
        <div className="batch-label"></div>

        <div className="info-text">
          Bury the Worries of Drawing the Molecules<br />
          A code by Dr. Nitin Sapre, Computational Chemist<br />
          Create Chemical Database with MOL files and SDF files using parallel processing.
        </div>

        <div className="molecules">
          {molecules.map((smiles, idx) => {
            let svg = ''
            if (rdkit) {
              try {
                const mol = rdkit.get_mol(smiles)
                if (mol) {
                  svg = mol.get_svg()
                }
              } catch (e) {
                console.error('Error rendering molecule:', e)
              }
            }
            return (
              <div key={idx} className="molecule">
                <div dangerouslySetInnerHTML={{ __html: svg }} />
                <p>{smiles}</p>
              </div>
            )
          })}
        </div>
      </div>

      {showAuthModal && (
        <div className="modal-overlay" onClick={() => setShowAuthModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>{isSignUp ? 'Sign Up' : 'Login'}</h2>
            <form onSubmit={isSignUp ? handleSignUp : handleLogin}>
              <input
                type="email"
                placeholder="Email"
                value={loginData.email}
                onChange={(e) => setLoginData(prev => ({ ...prev, email: e.target.value }))}
                required
              />
              <input
                type="password"
                placeholder="Password"
                value={loginData.password}
                onChange={(e) => setLoginData(prev => ({ ...prev, password: e.target.value }))}
                required
              />
              <button type="submit">{isSignUp ? 'Sign Up' : 'Login'}</button>
            </form>
            {!isSignUp && (
              <button onClick={handleForgotPassword} style={{ background: 'transparent', color: '#007bff', textDecoration: 'underline' }}>
                Forgot Password?
              </button>
            )}
            <button onClick={() => setIsSignUp(!isSignUp)}>
              {isSignUp ? 'Already have account? Login' : 'Need account? Sign Up'}
            </button>
            <button onClick={() => setShowAuthModal(false)}>Cancel</button>
          </div>
        </div>
      )}

      {toast && (
        <div className={`toast toast-${toast.type}`}>
          {toast.message}
        </div>
      )}
    </div>
  )
}

export default App
