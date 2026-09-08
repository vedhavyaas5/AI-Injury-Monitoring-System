import React, { useState } from 'react'
import LandingPage      from './pages/LandingPage'
import SelfAssessment   from './pages/SelfAssessment'
import MRIUpload        from './pages/MRIUpload'
import AssessmentResult from './pages/AssessmentResult'
import MRIResult        from './pages/MRIResult'

export default function App() {
  const [page,   setPage]   = useState('landing')   // landing | assess | mri | result | mriresult
  const [result, setResult] = useState(null)
  const [mriRes, setMriRes] = useState(null)

  const goto = (p) => setPage(p)

  if (page === 'landing')    return <LandingPage onAssess={() => goto('assess')}
                                                  onMRI={()    => goto('mri')} />
  if (page === 'assess')     return <SelfAssessment
                                      onBack={()  => goto('landing')}
                                      onDone={(r) => { setResult(r); goto('result') }} />
  if (page === 'mri')        return <MRIUpload
                                      onBack={()  => goto('landing')}
                                      onDone={(r) => { setMriRes(r); goto('mriresult') }} />
  if (page === 'result')     return <AssessmentResult
                                      data={result}
                                      onMRI={()   => goto('mri')}
                                      onHome={()  => goto('landing')} />
  if (page === 'mriresult')  return <MRIResult
                                      data={mriRes}
                                      onHome={()  => goto('landing')} />
  return null
}
