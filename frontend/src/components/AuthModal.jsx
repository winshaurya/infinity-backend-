import { useState } from 'react'

function AuthModal({ isOpen, onClose, supabase }) {
  const [loginData, setLoginData] = useState({ email: '', password: '' })
  const [isSignUp, setIsSignUp] = useState(false)
  const [toast, setToast] = useState(null)

  const showToast = (message, type = 'error') => {
    setToast({ message, type })
    setTimeout(() => setToast(null), 5000)
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
      onClose()
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
      onClose()
      setLoginData({ email: '', password: '' })
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

  if (!isOpen) return null

  return (
    <div className="modal-overlay" onClick={onClose}>
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
        <button onClick={onClose}>Cancel</button>
      </div>
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          {toast.message}
        </div>
      )}
    </div>
  )
}

export default AuthModal