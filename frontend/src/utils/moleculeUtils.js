export function updateFunctionalGroupStates(carbonCount) {
  const disabled = {}

  const disabledForZeroCarbon = new Set([
    'Amide', 'Cl', 'Br', 'I', 'F', 'CN', 'NC', 'OCN', 'NCO', 'Imine', 'NO2', 'Ketone', 'Ether', 'OH', 'NH2', 'OX_Cl', 'OX_Br', 'OX_F', 'OX_I', 'Azide',
    'S_Bivalent', 'S_Tetravalent', 'S_Hexavalent', 'S_Chain_Bi', 'S_Chain_Tetra', 'S_Chain_Hexa'
  ])

  const alwaysAllowed = new Set([
    'COOH', 'CHO', 'COOR_CH3', 'COOR_C2H5', 'COOR_C3H7', 'COOR_CH(CH3)2', 'COX_Cl', 'COX_Br', 'COX_F', 'COX_I'
  ])

  if (carbonCount === 0) {
    disabledForZeroCarbon.forEach(fg => {
      if (!alwaysAllowed.has(fg)) {
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

  return disabled
}

export function updateValencyStatus(carbonCount, doubleBonds, tripleBonds, rings, functionalGroups) {
  if (carbonCount === 0) {
    return 'Valency status: OK'
  }

  let maxValency = 2 * carbonCount + 2 - 2 * doubleBonds - 4 * tripleBonds
  if (rings > 0) {
    maxValency += rings
  }

  let currentValency = 0
  Object.values(functionalGroups).forEach(count => {
    currentValency += count
  })

  if (currentValency > maxValency) {
    return `Valency status: EXCEEDED (Current: ${currentValency}, Max: ${maxValency})`
  } else if (currentValency === maxValency) {
    return `Valency status: MAXIMUM REACHED (Current: ${currentValency}, Max: ${maxValency})`
  } else {
    return `Valency status: OK (Current: ${currentValency}, Max: ${maxValency})`
  }
}