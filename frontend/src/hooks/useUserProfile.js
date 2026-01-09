import { useState, useEffect } from 'react'
import { API_BASE } from '../constants.js'

export function useUserProfile(user, supabase) {
  const [profile, setProfile] = useState(null)

  const loadUserProfile = async () => {
    if (!user) return

    try {
      const token = (await supabase.auth.getSession()).data.session?.access_token
      const response = await fetch(`${API_BASE}/profile`, {
        headers: { 'Authorization': `Bearer ${token}` }
      })

      if (response.ok) {
        const data = await response.json()
        setProfile(data)
      }
    } catch (error) {
      console.error('Failed to load profile:', error)
    }
  }

  const refillCredits = async (amount) => {
    if (!user) return

    try {
      const token = (await supabase.auth.getSession()).data.session?.access_token
      const response = await fetch(`${API_BASE}/credits/refill`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ amount })
      })

      if (response.ok) {
        const data = await response.json()
        console.log('Credits refilled:', data.message)
        // Refresh profile to show updated credits
        await loadUserProfile()
        return { success: true, message: data.message, newBalance: data.new_balance }
      } else {
        const error = await response.json()
        console.error('Credit refill failed:', error)
        return { success: false, message: error.detail || 'Failed to refill credits' }
      }
    } catch (error) {
      console.error('Failed to refill credits:', error)
      return { success: false, message: 'Network error occurred' }
    }
  }

  useEffect(() => {
    if (user) {
      loadUserProfile()
    }
  }, [user])

  return {
    profile,
    loadUserProfile,
    refillCredits
  }
}