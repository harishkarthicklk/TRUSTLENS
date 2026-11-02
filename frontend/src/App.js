import React, { useState } from 'react';
import axios from 'axios';
import { Upload, Download, Play, CheckCircle2, AlertCircle, ShieldAlert, Tag } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api/v1';

export default function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('idle');
  const [logs, setLogs] = useState([]);
  const [pdfUrl, setPdfUrl] = useState(null);
  
  // Pipeline Analysis Results
  const [domain, setDomain] = useState(null);
  const [score1, setScore1] = useState(null);
  const [score2, setScore2] = useState(null);

  const addLog = (msg) =>
    setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  const handleFileUpload = (e) => {
    if (e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setPdfUrl(null);
      setLogs([]);
      setStatus('idle');
      setDomain(null);
      setScore1(null);
      setScore2(null);
    }
  };

  const runPipeline = async () => {
    if (!file) return;
    setStatus('running');
    setLogs([]);
    setPdfUrl(null);
    setDomain(null);
    setScore1(null);
    setScore2(null);

    try {
      // Phase 1: Ingestion & Preprocessing
      addLog('Phase 1: Ingesting dataset and running forensic preprocessing...');
      const formData = new FormData();
      formData.append('file', file);
      const p1Res = await axios.post(`${API_BASE}/phase1/upload`, formData);
      const preprocessedFile = p1Res.data.preprocessed_file;
      addLog(`Phase 1 Complete: Sanitized file stored as "${preprocessedFile}".`);

      // Phase 2: Domain Discovery & Test Selection
      addLog('Phase 2: Running domain discovery and AI test recommendation...');
      const p2Res = await axios.post(
        `${API_BASE}/phase2/discovery?filename=${encodeURIComponent(preprocessedFile)}`
      );
      
      const discoveryData = p2Res.data.data || {};
      const detectedDomain = discoveryData.classified_domain || discoveryData.domain || 'General Ledger / Financial Operations';
      setDomain(detectedDomain);

      const rawTests = discoveryData.recommended_tests || discoveryData.recommended_forensic_tests || [];
      const recommendedTests = rawTests.map((t) => (typeof t === 'object' ? t.test_name || t.name || JSON.stringify(t) : t));
      
      const finalTests = recommendedTests.length > 0 ? recommendedTests : [
        "Outlier Detection",
        "Correlation Analysis",
        "Duplicate Detection"
      ];
      addLog(`Phase 2 Complete: Domain identified as "${detectedDomain}". Selected tests -> ${finalTests.join(', ')}`);

      // Phase 3: Baseline Risk Scoring with Feature Relevance Filter
      addLog('Phase 3: Running dynamic feature selection & calculating baseline risk score (Score 1)...');
      const p3Res = await axios.post(`${API_BASE}/phase3/scoring`, {
        filename: preprocessedFile,
        threshold: 50.0,
        classified_domain: detectedDomain,
        recommended_tests: rawTests
      });
      
      const metrics = p3Res.data.metrics || {};
      const calculatedScore1 = metrics.score_1 ?? metrics.global_risk_score ?? 0;
      const p3FlaggedCount = metrics.p3_flagged_count ?? metrics.flagged_count ?? 0;
      const selectedFeatures = metrics.selected_features || [];
      setScore1(calculatedScore1);
      addLog(`Phase 3 Complete: Score 1 = ${calculatedScore1}/100 (${p3FlaggedCount} records flagged).`);
      addLog(`Features Selected for ML Engine -> [${selectedFeatures.join(', ')}]`);

      // Phase 4: Targeted Forensic Execution
      addLog('Phase 4: Executing forensic tests and calculating Score 2...');
      const p4Res = await axios.post(`${API_BASE}/phase4/execution`, {
        filename: preprocessedFile,
        selected_tests: finalTests
      });
      const calculatedScore2 = p4Res.data.score_2;
      setScore2(calculatedScore2);
      addLog(`Phase 4 Complete: Score 2 = ${calculatedScore2}/100.`);

      // Phase 5: LLM Summary & PDF Export
      addLog('Phase 5: Generating executive report and compiling PDF document...');
      const p5Res = await axios.post(`${API_BASE}/phase5/interpretation`, {
        filename: preprocessedFile,
        score_1: Number(calculatedScore1),
        score_2: Number(calculatedScore2),
        metrics: p4Res.data.summary,
        test_breakdown: p4Res.data.test_breakdown,
        sample_records: p4Res.data.high_risk_context_sample
      });

      setPdfUrl(`http://localhost:8000${p5Res.data.download_url}`);
      addLog('Pipeline completed successfully!');
      setStatus('completed');
    } catch (error) {
      let errorMsg = error.message;
      if (error.response?.data?.detail) {
        errorMsg = typeof error.response.data.detail === 'object'
          ? JSON.stringify(error.response.data.detail)
          : error.response.data.detail;
      }
      addLog(`Pipeline Error: ${errorMsg}`);
      setStatus('failed');
    }
  };

  return (
    <div style={{ maxWidth: '850px', margin: '40px auto', fontFamily: 'system-ui, -apple-system, sans-serif', padding: '24px' }}>
      <h2 style={{ color: '#1A365D', marginBottom: '20px' }}>TrustLens Forensic Audit Pipeline</h2>

      {/* File Dropzone */}
      <div style={{ border: '2px dashed #CBD5E0', padding: '30px', borderRadius: '8px', textAlign: 'center', backgroundColor: '#F7FAFC' }}>
        <Upload size={40} color="#4A5568" style={{ marginBottom: '10px' }} />
        <br />
        <input type="file" accept=".csv, .xlsx, .xls" onChange={handleFileUpload} />
        {file && <p style={{ marginTop: '12px', color: '#2D3748' }}>Selected Dataset: <b>{file.name}</b></p>}
      </div>

      <br />

      {/* Action Button */}
      <button
        onClick={runPipeline}
        disabled={!file || status === 'running'}
        style={{
          backgroundColor: file && status !== 'running' ? '#2B6CB0' : '#A0AEC0',
          color: 'white',
          padding: '12px 24px',
          border: 'none',
          borderRadius: '6px',
          cursor: file && status !== 'running' ? 'pointer' : 'not-allowed',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          fontWeight: '600',
          fontSize: '15px'
        }}
      >
        <Play size={18} /> {status === 'running' ? 'Processing Pipeline...' : 'Run Full Audit Pipeline'}
      </button>

      {/* Results Summary Dashboard Card */}
      {(domain || score1 !== null || score2 !== null) && (
        <div style={{ marginTop: '24px', padding: '20px', backgroundColor: '#EDF2F7', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
          <h3 style={{ margin: '0 0 16px 0', color: '#2D3748', fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldAlert size={20} color="#2B6CB0" /> Dataset Audit Summary
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
            {/* Domain Classification */}
            <div style={{ backgroundColor: 'white', padding: '16px', borderRadius: '6px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#718096', fontSize: '13px', fontWeight: 'bold', textTransform: 'uppercase' }}>
                <Tag size={14} /> Classified Domain
              </div>
              <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#2B6CB0', marginTop: '6px' }}>
                {domain || 'Detecting...'}
              </div>
            </div>

            {/* Baseline Risk Score */}
            <div style={{ backgroundColor: 'white', padding: '16px', borderRadius: '6px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
              <div style={{ color: '#718096', fontSize: '13px', fontWeight: 'bold', textTransform: 'uppercase' }}>
                Phase 3 Risk Score
              </div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#DD6B20', marginTop: '6px' }}>
                {score1 !== null ? `${score1} / 100` : 'Calculating...'}
              </div>
            </div>

            {/* Targeted Forensic Score */}
            <div style={{ backgroundColor: 'white', padding: '16px', borderRadius: '6px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
              <div style={{ color: '#718096', fontSize: '13px', fontWeight: 'bold', textTransform: 'uppercase' }}>
                Phase 4 Forensic Score
              </div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', color: '#E53E3E', marginTop: '6px' }}>
                {score2 !== null ? `${score2} / 100` : 'Calculating...'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Execution Logs */}
      <div style={{ marginTop: '24px', backgroundColor: '#1A202C', color: '#68D391', padding: '20px', borderRadius: '8px', fontFamily: 'monospace', fontSize: '13.5px', minHeight: '180px', overflowY: 'auto' }}>
        <h4 style={{ color: '#E2E8F0', marginTop: 0, marginBottom: '12px' }}>Execution Logs:</h4>
        {logs.length === 0 ? (
          <p style={{ color: '#A0AEC0' }}>Select a CSV or Excel file and start the pipeline to view real-time operations.</p>
        ) : (
          logs.map((log, idx) => <p key={idx} style={{ margin: '4px 0' }}>{log}</p>)
        )}
      </div>

      {/* Status Indicators & Download Option */}
      {status === 'completed' && pdfUrl && (
        <div style={{ marginTop: '24px', padding: '16px', backgroundColor: '#F0FFF4', border: '1px solid #C6F6D5', borderRadius: '8px', textAlign: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', color: '#22543D', marginBottom: '12px', fontWeight: 'bold' }}>
            <CheckCircle2 color="#38A169" size={20} /> Audit Pipeline Execution Completed
          </div>
          <a
            href={pdfUrl}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              backgroundColor: '#38A169',
              color: 'white',
              padding: '12px 24px',
              borderRadius: '6px',
              textDecoration: 'none',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              fontWeight: 'bold'
            }}
          >
            <Download size={18} /> Download Generated Audit PDF
          </a>
        </div>
      )}

      {status === 'failed' && (
        <div style={{ marginTop: '24px', padding: '16px', backgroundColor: '#FFF5F5', border: '1px solid #FEB2B2', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '10px', color: '#9B2C2C' }}>
          <AlertCircle size={20} color="#E53E3E" /> Pipeline execution halted due to an error. Check logs above.
        </div>
      )}
    </div>
  );
}