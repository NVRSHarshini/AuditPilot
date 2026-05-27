/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { 
  ShieldAlert, 
  ShieldCheck, 
  ArrowRight, 
  CheckCircle2, 
  FileText, 
  UploadCloud, 
  Copy, 
  Send, 
  Users, 
  Briefcase, 
  AlertOctagon, 
  RefreshCw, 
  ExternalLink,
  Check,
  Code,
  FileCheck,
  ChevronRight,
  Sparkles
} from "lucide-react";
import { Gap, PolicyControl, ChecklistRequirement } from "./types";

async function safeJson(res: Response): Promise<any> {
  const contentType = res.headers.get("content-type");
  if (!contentType || !contentType.includes("application/json")) {
    const text = await res.text();
    if (text.includes("<!doctype") || text.includes("<html")) {
      throw new Error("Server returned an HTML page. The backend server might still be compiling or starting up, please try again in a few seconds.");
    }
    throw new Error(`Non-JSON response from server (Status: ${res.status}): ${text.substring(0, 100)}`);
  }
  return res.json();
}

function parseInline(text: string): React.ReactNode[] {
  const tokens: React.ReactNode[] = [];
  let remaining = text;
  
  while (remaining) {
    const boldMatch = remaining.match(/\*\*(.*?)\*\*/);
    const codeMatch = remaining.match(/`(.*?)`/);
    
    const boldIndex = boldMatch ? remaining.indexOf(boldMatch[0]) : -1;
    const codeIndex = codeMatch ? remaining.indexOf(codeMatch[0]) : -1;
    
    if (boldIndex === -1 && codeIndex === -1) {
      tokens.push(remaining);
      break;
    }
    
    const isBoldFirst = boldIndex !== -1 && (codeIndex === -1 || boldIndex < codeIndex);
    
    if (isBoldFirst && boldMatch) {
      if (boldIndex > 0) {
        tokens.push(remaining.substring(0, boldIndex));
      }
      tokens.push(<strong key={remaining.substring(0, 5) + "-bold-" + boldIndex} className="font-extrabold text-slate-900">{boldMatch[1]}</strong>);
      remaining = remaining.substring(boldIndex + boldMatch[0].length);
    } else if (codeMatch) {
      if (codeIndex > 0) {
        tokens.push(remaining.substring(0, codeIndex));
      }
      tokens.push(<code key={remaining.substring(0, 5) + "-code-" + codeIndex} className="bg-slate-100 text-indigo-700 px-1.5 py-0.5 rounded font-mono text-[10px] border border-slate-200">{codeMatch[1]}</code>);
      remaining = remaining.substring(codeIndex + codeMatch[0].length);
    }
  }
  return tokens.length > 0 ? tokens : [text];
}

function SimpleMarkdown({ text }: { text: string }) {
  if (!text) return null;
  
  const lines = text.split("\n");
  const blocks: React.ReactNode[] = [];
  
  let currentList: { items: string[]; ordered: boolean } | null = null;
  let currentTable: { headers: string[]; rows: string[][] } | null = null;
  
  const flushList = (key: string) => {
    if (currentList) {
      const listItems = currentList.items.map((item, idx) => (
        <li key={idx} className="mb-1 text-slate-700 list-disc ml-4 pl-1">
          {parseInline(item)}
        </li>
      ));
      blocks.push(
        <ul key={`list-${key}`} className="my-2 space-y-1">
          {listItems}
        </ul>
      );
      currentList = null;
    }
  };
  
  const flushTable = (key: string) => {
    if (currentTable) {
      blocks.push(
        <div key={`table-${key}`} className="my-3 overflow-x-auto border border-slate-200 rounded-lg">
          <table className="min-w-full divide-y divide-slate-200 text-xs">
            <thead className="bg-slate-50">
              <tr>
                {currentTable.headers.map((h, i) => (
                  <th key={i} className="px-3 py-2 text-left font-bold text-slate-700 uppercase tracking-wider">
                    {parseInline(h)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-slate-200">
              {currentTable.rows.map((row, rIdx) => (
                <tr key={rIdx} className={rIdx % 2 === 0 ? "bg-white" : "bg-slate-50/50"}>
                  {row.map((cell, cIdx) => (
                    <td key={cIdx} className="px-3 py-1.5 text-slate-650 max-w-xs whitespace-normal">
                      {parseInline(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      currentTable = null;
    }
  };
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    if (line === "---" || line === "***" || line === "--- ") {
      flushList(`hr-${i}`);
      flushTable(`hr-${i}`);
      blocks.push(<hr key={i} className="my-4 border-slate-200" />);
      continue;
    }
    
    if (line.startsWith("# ")) {
      flushList(`h1-${i}`);
      flushTable(`h1-${i}`);
      blocks.push(<h2 key={i} className="text-sm font-bold text-slate-900 mt-5 mb-2 pb-1 border-b border-slate-100">{parseInline(line.replace("# ", ""))}</h2>);
      continue;
    }
    if (line.startsWith("## ")) {
      flushList(`h2-${i}`);
      flushTable(`h2-${i}`);
      blocks.push(<h3 key={i} className="text-xs font-bold text-[#1F3A5F] mt-4 mb-2 uppercase tracking-wide">{parseInline(line.replace("## ", ""))}</h3>);
      continue;
    }
    if (line.startsWith("### ")) {
      flushList(`h3-${i}`);
      flushTable(`h3-${i}`);
      blocks.push(<h4 key={i} className="text-[11px] font-bold text-slate-800 mt-3 mb-1.5">{parseInline(line.replace("### ", ""))}</h4>);
      continue;
    }
    
    if (line.startsWith("|")) {
      flushList(`table-${i}`);
      const cells = line.split("|").map(c => c.trim()).filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
      
      if (cells.every(c => c.startsWith("-"))) {
        continue;
      }
      
      if (!currentTable) {
        currentTable = { headers: cells, rows: [] };
      } else {
        currentTable.rows.push(cells);
      }
      continue;
    } else {
      flushTable(`text-${i}`);
    }
    
    if (line.startsWith("- ") || line.startsWith("* ")) {
      const content = line.substring(2);
      if (!currentList) {
        currentList = { items: [content], ordered: false };
      } else {
        currentList.items.push(content);
      }
      continue;
    } else {
      flushList(`text-${i}`);
    }
    
    if (line !== "") {
      blocks.push(<p key={i} className="mb-2 leading-relaxed text-slate-650 text-xs">{parseInline(line)}</p>);
    }
  }
  
  flushList("final");
  flushTable("final");
  
  return <div className="space-y-1">{blocks}</div>;
}

function PdfPreview({ file, type }: { file: File; type: "policy" | "checklist" }) {
  const [previewText, setPreviewText] = React.useState<string>("Extracting compliance contents...");

  React.useEffect(() => {
    if (!file) return;
    
    let active = true;
    let retryCount = 0;
    const maxRetries = 5;
    const delayMs = 1500;

    const fetchPreview = () => {
      if (!active) return;
      setPreviewText(retryCount > 0 ? `Extracting compliance contents (Attempting connection ${retryCount}/${maxRetries})...` : "Extracting compliance contents...");

      const formData = new FormData();
      formData.append("file", file);
      formData.append("type", type);

      fetch("/api/pdf/preview", {
        method: "POST",
        body: formData,
      })
        .then((res) => {
          return safeJson(res);
        })
        .then((data) => {
          if (active) {
            setPreviewText(data.text || "No preview text extracted.");
          }
        })
        .catch((err: Error) => {
          if (!active) return;
          console.warn(`PDF preview fetch failed (attempt ${retryCount}/${maxRetries}):`, err.message || err);
          
          if (retryCount < maxRetries) {
            retryCount++;
            setTimeout(fetchPreview, delayMs);
          } else {
            setPreviewText(`Could not load preview: ${err.message || "Failed to fetch from server."}`);
          }
        });
    };

    fetchPreview();

    return () => {
      active = false;
    };
  }, [file, type]);

  const formattedSize = file.size < 1024 * 1024
    ? `${(file.size / 1024).toFixed(1)} KB`
    : `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

  return (
    <div className="mt-3.5 border border-slate-200 rounded-lg overflow-hidden bg-slate-50 text-left w-full shadow-xs">
      <div className="bg-slate-100 px-3 py-1.5 border-b border-slate-200 flex justify-between items-center">
        <span className="text-[10px] font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
          <FileText className="w-3.5 h-3.5 text-[#1F3A5F]" /> {type === "policy" ? "Corporate Policy Document Preview" : "ISO 27001 Checklist Preview"}
        </span>
        <span className="text-[8px] bg-emerald-100 text-emerald-800 px-1.5 py-0.5 rounded-full font-bold flex items-center gap-1">
          <Check className="w-2.5 h-2.5 text-emerald-600" />
          File Ready ✓
        </span>
      </div>
      <div className="p-3">
        <div className="flex justify-between items-center gap-2">
          <div className="min-w-0 flex-1">
            <p className="text-xs font-bold text-slate-700 truncate">{file.name}</p>
            <p className="text-[10px] text-slate-500 font-mono">{formattedSize}</p>
          </div>
          <div className="bg-emerald-50 rounded-full p-1 border border-emerald-200 shrink-0">
            <Check className="w-3.5 h-3.5 text-emerald-600" />
          </div>
        </div>

        <div className="mt-2.5 bg-slate-100 p-2 rounded border border-slate-200">
          <p className="text-[9px] text-[#1F3A5F] font-bold uppercase tracking-wider mb-1">
            Content Preview:
          </p>
          <pre className="text-[10px] text-slate-600 font-mono leading-relaxed max-h-24 overflow-y-auto whitespace-pre-wrap select-text">
            {previewText}
          </pre>
        </div>
      </div>
    </div>
  );
}

interface AuditResult {
  session_id: string;
  status: string;
  gaps: Gap[];
  high_risk_gaps: Gap[];
  report: string | null;
}

export default function App() {
  const [stage, setStage] = useState<1 | 2 | 3 | 4>(1);
  const [policyFile, setPolicyFile] = useState<File | null>(null);
  const [checklistFile, setChecklistFile] = useState<File | null>(null);
  const [isDraggingPolicy, setIsDraggingPolicy] = useState(false);
  const [isDraggingChecklist, setIsDraggingChecklist] = useState(false);
  const [sampleMode, setSampleMode] = useState(false);
  
  // Running simulation / feedback status
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [sessionId, setSessionId] = useState<string>("");
  const [gaps, setGaps] = useState<Gap[]>([]);
  const [highRiskGaps, setHighRiskGaps] = useState<Gap[]>([]);
  const [finalReport, setFinalReport] = useState<string>("");
  const [decisions, setDecisions] = useState<Record<string, "escalate" | "accept_risk">>({});
  const [error, setError] = useState<string | null>(null);
  const [isSubmittingDecisions, setIsSubmittingDecisions] = useState(false);
  const [copied, setCopied] = useState(false);
  const [dbConnectedMessage, setDbConnectedMessage] = useState<string>("");

  // Hidden file inputs
  const policyInputRef = useRef<HTMLInputElement>(null);
  const checklistInputRef = useRef<HTMLInputElement>(null);

  const pipelineSteps = [
    { label: "Extracting controls", desc: "Using Gemini 3.5 Flash to parse and catalog custom corporate controls" },
    { label: "Storing in Database", desc: "Generating vector embeddings (768-dim) and indexing in compliance_controls" },
    { label: "Detecting compliance gaps", desc: "Running cosine similarity search to identify missing security policy layers" },
    { label: "Scoring gaps with risks", desc: "AI evaluating operational breach risks and mitigation difficulties" },
    { label: "Assigning roadmap owners", desc: "Routing deficiencies to specialized IT Security and SOC departments" }
  ];

  // Load sample PDFs
  const loadSamples = () => {
    const samplePolicyText = `ACME Security Policy
ACME Corp Information Security Policy v2.1

1. ACCESS CONTROL
All users must use passwords of at least 8 characters.
MFA is recommended but not mandatory for internal systems.
Privileged accounts must be reviewed quarterly.

2. DATA PROTECTION
Sensitive data must be encrypted at rest using AES-128.
Data in transit must use TLS 1.1 or higher.
Customer PII must not be stored on local machines.

3. INCIDENT RESPONSE
Security incidents must be reported to IT within 48 hours.
Incident logs must be retained for 6 months.
A post-incident review must be conducted within 2 weeks.

4. BACKUP AND RECOVERY
System backups are performed weekly every Sunday.
Backup restoration is tested once per year.

5. VENDOR MANAGEMENT
Third party vendors must sign an NDA before data access.
Vendor access must be reviewed annually.`;

    const sampleChecklistText = `ISO 27001 Checklist
ISO 27001 Compliance Checklist 2024

ACCESS CONTROL REQUIREMENTS
AC-1: MFA MUST be enforced for ALL user accounts including standard users. [REQUIRED]
AC-2: Minimum password length must be 12 characters or more. [REQUIRED]
AC-3: Privileged account access must be reviewed monthly. [REQUIRED]

DATA PROTECTION REQUIREMENTS
DP-1: Data at rest must be encrypted using AES-256 or stronger. [REQUIRED]
DP-2: Data in transit must use TLS 1.2 or higher. [REQUIRED]
DP-3: PII data must be masked in all non-production environments. [REQUIRED]

INCIDENT RESPONSE REQUIREMENTS
IR-1: Security incidents must be reported within 24 hours. [REQUIRED]
IR-2: Incident logs must be retained for minimum 12 months. [REQUIRED]

BACKUP REQUIREMENTS
BC-1: Backups must be performed daily. [REQUIRED]
BC-2: Backup restoration must be tested quarterly. [REQUIRED]

VENDOR REQUIREMENTS
VM-1: Vendors must complete security assessment before access. [REQUIRED]
VM-2: Vendor access must be reviewed every 6 months. [REQUIRED]`;

    const samplePolicy = new File([samplePolicyText], "ACME_Security_Policy_v2.1.pdf", { type: "application/pdf" });
    const sampleChecklist = new File([sampleChecklistText], "ISO27001_Compliance_Checklist.pdf", { type: "application/pdf" });
    setPolicyFile(samplePolicy);
    setChecklistFile(sampleChecklist);
    setSampleMode(true);
    setError(null);
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent, type: "policy" | "checklist") => {
    e.preventDefault();
    if (type === "policy") setIsDraggingPolicy(true);
    if (type === "checklist") setIsDraggingChecklist(true);
  };

  const handleDragLeave = (type: "policy" | "checklist") => {
    if (type === "policy") setIsDraggingPolicy(false);
    if (type === "checklist") setIsDraggingChecklist(false);
  };

  const handleDrop = (e: React.DragEvent, type: "policy" | "checklist") => {
    e.preventDefault();
    handleDragLeave(type);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type === "application/pdf" || file.name.endsWith(".pdf")) {
        if (type === "policy") setPolicyFile(file);
        if (type === "checklist") setChecklistFile(file);
        setSampleMode(false);
        setError(null);
      } else {
        setError("Please upload PDF documents only.");
      }
    }
  };

  // Run audit agent handler
  const runAudit = async () => {
    if (!policyFile || !checklistFile) {
      setError("Please select both files before running the audit.");
      return;
    }

    setError(null);
    setDbConnectedMessage("");

    // Test MongoDB connection beforehand
    try {
      const pingRes = await fetch("/api/db/test");
      const pingData = await safeJson(pingRes);
      if (!pingRes.ok || !pingData.success) {
        throw new Error(pingData.error || "MongoDB Atlas connection failed — check your connection string");
      }
      setDbConnectedMessage("MongoDB Atlas Connected ✓");
    } catch (pingErr: any) {
      setError(pingErr.message || "MongoDB Atlas connection failed — check your connection string");
      return; // Do not proceed
    }

    setStage(2);
    setCurrentStepIndex(0);

    const formData = new FormData();
    formData.append("policy_file", policyFile);
    formData.append("checklist_file", checklistFile);
    if (sampleMode) {
      formData.append("sample_mode", "true");
    }

    // Set up holders for concurrent fetch state
    let fetchResult: AuditResult | null = null;
    let fetchError: Error | null = null;

    const startFetch = async () => {
      try {
        const res = await fetch("/api/audit/start", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const errData = await safeJson(res).catch(() => ({ error: `Failed with status ${res.status}` }));
          throw new Error(errData.error || "Failed while analyzing compliance PDFs.");
        }

        const data: AuditResult = await safeJson(res);
        fetchResult = data;
      } catch (err: any) {
        fetchError = err;
      }
    };

    // Execute API fetch as a background promise
    const fetchPromise = startFetch();

    try {
      // Step-by-step progressive visual sequence (simulating step index 0 to 4)
      const stepDurationMs = 1200;
      for (let step = 0; step < 5; step++) {
        setCurrentStepIndex(step);
        await new Promise(resolve => setTimeout(resolve, stepDurationMs));
      }

      // After completing index 4, wait for the actual background backend call to finish
      await fetchPromise;

      if (fetchError) {
        throw fetchError;
      }

      if (!fetchResult) {
        throw new Error("No audit data received from server.");
      }

      setSessionId(fetchResult.session_id);
      setGaps(fetchResult.gaps || []);
      setHighRiskGaps(fetchResult.high_risk_gaps || []);

      if (fetchResult.status === "awaiting_approval") {
        setStage(3);
        // Pre-initialize decisions mapping for each gap as undefined to force reviewer to choose
        setDecisions({});
      } else {
        setFinalReport(fetchResult.report || "");
        setStage(4);
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred during processing.");
      setStage(1);
    }
  };

  // Choose decision handler
  const selectDecision = (controlId: string, choice: "escalate" | "accept_risk") => {
    setDecisions(prev => ({
      ...prev,
      [controlId]: choice
    }));
  };

  // Submit human decisions to resume pipeline
  const submitDecisions = async () => {
    // Verify all high risk gaps have decisions
    const totalHighRisk = highRiskGaps.length;
    const handledTotal = Object.keys(decisions).length;
    if (handledTotal < totalHighRisk) return;

    setIsSubmittingDecisions(true);
    setError(null);

    try {
      const res = await fetch("/api/audit/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          decisions
        })
      });

      if (!res.ok) {
        const errData = await safeJson(res).catch(() => ({ error: `Failed with status ${res.status}` }));
        throw new Error(errData.error || "Failed while updating checkpoint approval.");
      }

      const data = await safeJson(res);
      setFinalReport(data.report || "");
      
      // Update gaps array with human decision properties so Stage 4 can render them
      setGaps(prev => prev.map(gap => ({
        ...gap,
        human_decision: decisions[gap.control_id] || gap.human_decision
      })));

      setStage(4);
    } catch (err: any) {
      setError(err.message || "Failed when finalizing human vetting checkpoint.");
    } finally {
      setIsSubmittingDecisions(false);
    }
  };

  // Copy report as JSON
  const copyReportJson = () => {
    const backupJson = JSON.stringify({
      session_id: sessionId,
      total_gaps: gaps.length,
      overall_risk: getGradeBadge(gaps).text,
      compliance_gaps: gaps,
      auditor_report: finalReport
    }, null, 2);
    
    navigator.clipboard.writeText(backupJson);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Download compliance audit report as HTML (works in full sandboxes/iframes)
  const downloadReportHtml = () => {
    const simpleMarkdownToHtml = (md: string) => {
      if (!md) return "";
      let html = md
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
      
      html = html.replace(/^### (.*$)/gim, '<h3 style="color: #1E3A8A; margin-top: 1.5em; margin-bottom: 0.5em; font-size: 1.25rem; font-weight: 700; border-bottom: 1px solid #E2E8F0; padding-bottom: 0.3em;">$1</h3>');
      html = html.replace(/^## (.*$)/gim, '<h2 style="color: #1F3A5F; margin-top: 2em; margin-bottom: 0.6em; font-size: 1.5rem; font-weight: 800; border-bottom: 2px solid #CBD5E1; padding-bottom: 0.4em;">$1</h2>');
      html = html.replace(/^# (.*$)/gim, '<h1 style="color: #0F172A; margin-top: 0; margin-bottom: 0.8em; font-size: 2rem; font-weight: 900; background: #F1F5F9; padding: 0.75em; border-radius: 8px;">$1</h1>');
      html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      html = html.replace(/^\s*[-*]\s+(.*$)/gim, '<li style="margin-left: 20px; margin-bottom: 0.5em;">$1</li>');
      
      const lines = html.split("\n");
      let insideList = false;
      const processedLines = lines.map(line => {
        const isLi = line.trim().startsWith("<li");
        if (isLi && !insideList) {
          insideList = true;
          return '<ul style="margin-top: 0.5em; margin-bottom: 1rem; padding-left: 1.2em;">' + line;
        } else if (!isLi && insideList) {
          insideList = false;
          return "</ul>" + (line.trim() ? `<p style="margin-bottom: 1rem; line-height: 1.6; color: #334155;">${line}</p>` : line);
        }
        if (!isLi && line.trim() && !line.startsWith("<h") && !line.startsWith("<ul") && !line.startsWith("</ul")) {
          return `<p style="margin-bottom: 1rem; line-height: 1.6; color: #334155;">${line}</p>`;
        }
        return line;
      });
      return processedLines.join("\n");
    };

    const gapsRows = gaps.map(gap => {
      const riskClass = gap.risk_level === "Critical" ? "risk-Critical" : gap.risk_level === "High" ? "risk-High" : gap.risk_level === "Medium" ? "risk-Medium" : "risk-Low";
      const decisionText = gap.human_decision === "escalate" 
        ? "Action Plan Escalated" 
        : gap.human_decision === "accept_risk" 
          ? "Risk Accepted" 
          : "Remediation Pending";
      const decisionBadgeClass = gap.human_decision === "escalate" ? "badge-approved" : gap.human_decision === "accept_risk" ? "badge-rejected" : "";
      
      return `
        <tr>
          <td style="font-weight: bold; font-family: monospace; white-space: nowrap; color: #0F172A; font-size: 0.85rem; border-bottom: 1px solid #E2E8F0;">${gap.control_id}</td>
          <td style="border-bottom: 1px solid #E2E8F0;">
            <div style="font-weight: 600; color: #1E293B; margin-bottom: 6px; font-size: 0.95rem;">${gap.description}</div>
            <div class="gap-detail-box">
              <p style="margin: 0 0 6px 0; color: #475569; line-height: 1.4;"><strong style="color: #64748B;">Reason:</strong> ${gap.gap_reason}</p>
              <p style="margin: 0; color: #475569; line-height: 1.4;"><strong style="color: #1E3A8A;">Remediation Action:</strong> ${gap.recommended_action}</p>
            </div>
          </td>
          <td style="border-bottom: 1px solid #E2E8F0;"><span class="risk-badge ${riskClass}">${gap.risk_level}</span></td>
          <td style="white-space: nowrap; font-weight: 500; font-size: 0.85rem; color: #475569; border-bottom: 1px solid #E2E8F0;">${gap.owner}</td>
          <td style="border-bottom: 1px solid #E2E8F0;">
            <span class="risk-badge ${decisionBadgeClass}" style="${!gap.human_decision ? "background:#F3F4F6; color:#4B5563; border:1px solid #E5E7EB;" : ""}">${decisionText}</span>
          </td>
        </tr>
      `;
    }).join("");

    const htmlDoc = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AuditPilot Compliance Executive Report</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      color: #334155;
      background-color: #F8FAFC;
      margin: 0;
      padding: 0;
      line-height: 1.5;
    }
    .container {
      max-width: 1000px;
      margin: 40px auto;
      background: white;
      border-radius: 12px;
      box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -4px rgba(0,0,0,0.1);
      border: 1px solid #E2E8F0;
      overflow: hidden;
    }
    .header {
      background-color: #1F3A5F;
      color: white;
      padding: 40px;
    }
    .header h1 {
      margin: 0;
      font-size: 2.25rem;
      font-weight: 800;
      letter-spacing: -0.025em;
    }
    .header p {
      margin: 10px 0 0 0;
      opacity: 0.9;
      font-size: 1.05rem;
    }
    .meta-bar {
      background-color: #0F172A;
      color: #94A3B8;
      padding: 15px 40px;
      font-size: 0.85rem;
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 15px;
    }
    .meta-val {
      color: white;
      font-weight: 600;
      font-family: monospace;
    }
    .content {
      padding: 40px;
    }
    .risk-badge {
      display: inline-block;
      padding: 4px 10px;
      font-size: 10px;
      font-weight: 700;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .risk-High {
      background-color: #FEE2E2;
      color: #991B1B;
      border: 1px solid #FCA5A5;
    }
    .risk-Medium {
      background-color: #FFEDD5;
      color: #9A3412;
      border: 1px solid #FDBA74;
    }
    .risk-Low {
      background-color: #FEF9C3;
      color: #713F12;
      border: 1px solid #FDE047;
    }
    .badge-approved {
      background-color: #DCFCE7;
      color: #166534;
      border: 1px solid #86EFAC;
    }
    .badge-rejected {
      background-color: #F3F4F6;
      color: #374151;
      border: 1px solid #D1D5DB;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 25px;
      margin-bottom: 25px;
      font-size: 0.9rem;
    }
    th {
      background-color: #F1F5F9;
      color: #1E293B;
      text-align: left;
      font-weight: 700;
      padding: 12px 16px;
      border-bottom: 2px solid #E2E8F0;
    }
    td {
      padding: 14px 16px;
      vertical-align: top;
    }
    .section-title {
      font-size: 1.5rem;
      font-weight: 700;
      color: #1E293B;
      margin-top: 2rem;
      margin-bottom: 1rem;
      border-bottom: 2px solid #E2E8F0;
      padding-bottom: 8px;
    }
    .gap-detail-box {
      background-color: #F8FAFC;
      border: 1px solid #E2E8F0;
      border-radius: 6px;
      padding: 12px;
      margin-top: 8px;
      font-size: 0.85rem;
    }
    footer {
      text-align: center;
      color: #94A3B8;
      font-size: 0.8rem;
      padding: 30px 40px;
      border-top: 1px solid #E2E8F0;
      background-color: #F8FAFC;
    }
    @media print {
      body {
        background-color: white;
      }
      .container {
        box-shadow: none;
        border: none;
        margin: 0;
        max-width: 100%;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>AuditPilot Compliance Executive Report</h1>
      <p>Automated standard cross-gapping assessment with Multi-Agent verification workflows.</p>
    </div>
    <div class="meta-bar">
      <div>Session ID: <span class="meta-val">${sessionId}</span></div>
      <div>Compliance Gaps: <span class="meta-val" style="color: #EF4444;">${gaps.length} Found</span></div>
      <div>Assessed Date: <span class="meta-val">${new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</span></div>
    </div>
    <div class="content">
      <div class="section-title">I. Executive Summary Report</div>
      <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 25px; border-radius: 8px; margin-bottom: 30px;">
        ${simpleMarkdownToHtml(finalReport)}
      </div>

      <div class="section-title">II. Detailed Compliance Gaps & Actions</div>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Requirement & Description</th>
              <th>Risk Level</th>
              <th>Owner Route</th>
              <th>Status / Decision</th>
            </tr>
          </thead>
          <tbody>
            ${gapsRows || '<tr><td colspan="5" style="text-align:center; padding: 20px; color:#64748B;">No Gaps Found. Fully Compliant!</td></tr>'}
          </tbody>
        </table>
      </div>
    </div>
    <footer>
      <p>AuditPilot Standard Corporate Auditor • Confidential Internal Compliance Documentation</p>
      <p>&copy; ${new Date().getFullYear()} Enterprise Governance. Powered by Google Gemini.</p>
    </footer>
  </div>
</body>
</html>`;

    const blob = new Blob([htmlDoc], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `AuditPilot_Report_${sessionId || "session"}.html`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Grade calculators
  const getGradeBadge = (allGaps: Gap[]) => {
    const high = allGaps.filter(g => g.risk_level === "High" || g.risk_level === "Critical").length;
    const med = allGaps.filter(g => g.risk_level === "Medium").length;
    if (high > 0) return { text: "Critical Risk Found", bg: "bg-red-50 text-red-700 border-red-200", colorClass: "text-red-700" };
    if (med > 0) return { text: "High Security Findings", bg: "bg-orange-50 text-orange-700 border-orange-200", colorClass: "text-orange-700" };
    return { text: "Low Action Required", bg: "bg-emerald-50 text-emerald-700 border-emerald-200", colorClass: "text-emerald-700" };
  };

  const getRiskColor = (level: "Critical" | "High" | "Medium" | "Low") => {
    if (level === "Critical") return { card: "border-l-4 border-l-purple-600", text: "text-purple-700", bg: "bg-purple-50 text-purple-700 border-purple-100" };
    if (level === "High") return { card: "border-l-4 border-l-red-500", text: "text-red-700", bg: "bg-red-50 text-red-700 border-red-100" };
    if (level === "Medium") return { card: "border-l-4 border-l-orange-500", text: "text-orange-700", bg: "bg-orange-50 text-orange-700 border-orange-100" };
    return { card: "border-l-4 border-l-yellow-500", text: "text-yellow-700", bg: "bg-yellow-50 text-yellow-700 border-yellow-101" };
  };

  return (
    <>
      <div className="h-screen w-full bg-[#f8fafc] flex flex-col font-sans overflow-hidden print:hidden">
      {/* High Density Header */}
      <header className="h-16 bg-[#1F3A5F] text-white px-6 flex items-center justify-between border-b border-slate-700 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-400 rounded-md flex items-center justify-center font-bold text-[#1F3A5F] shrink-0 font-sans">
            AP
          </div>
          <h1 className="text-xl font-semibold tracking-tight">
            AuditPilot
          </h1>
        </div>

        <div className="flex items-center gap-6 text-xs font-medium">
          {sampleMode && (
            <div className="flex items-center gap-1.5 bg-amber-500/20 text-amber-300 px-2.5 py-1 rounded border border-amber-500/40 font-semibold animate-fade-in">
              <Sparkles className="w-3.5 h-3.5 text-amber-400 shrink-0 animate-pulse" />
              <span>Demo Sample Mode</span>
            </div>
          )}
          <div className="flex items-center gap-2 bg-blue-900/50 px-3 py-1 rounded border border-blue-800">
            <span className="text-blue-300">Session ID:</span>
            <span className="font-mono">{sessionId || "#82910-XZ-2024"}</span>
          </div>
          <div className="flex gap-2 items-center">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
            <span className="text-slate-300 italic">Agent Online: Gemini-3.5-Flash</span>
          </div>
          {stage > 1 && (
            <button 
              onClick={() => {
                setStage(1);
                setPolicyFile(null);
                setChecklistFile(null);
                setGaps([]);
                setHighRiskGaps([]);
                setDecisions({});
                setSessionId("");
                setSampleMode(false);
              }}
              className="flex items-center gap-1.5 px-2.5 py-1 bg-blue-900/40 hover:bg-blue-800/60 border border-blue-700 rounded text-slate-200 transition-all font-medium cursor-pointer"
            >
              <RefreshCw className="w-3 h-3" /> Start New Audit
            </button>
          )}
        </div>
      </header>

      {/* Main Structural Layout Split (Two-pane layout) */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar Layout */}
        <aside className="w-64 bg-white border-r border-slate-200 flex flex-col shrink-0 hidden lg:flex select-none">
          <div className="p-4 border-b border-slate-100 flex-1 overflow-y-auto">
            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Audit Pipeline</h2>
            
            <div className="space-y-4">
              {/* Parse Assets Step */}
              <div className="flex gap-3 items-start">
                <div className={`mt-0.5 w-5 h-5 rounded-full flex items-center justify-center text-white text-[10px] shrink-0 ${
                  stage > 1 ? "bg-green-500" : stage === 1 ? "bg-indigo-600" : "bg-green-500"
                }`}>
                  {stage > 1 ? "✓" : "1"}
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-800">Parse Assets</p>
                  <p className="text-[10px] text-slate-400 leading-snug">
                    {stage > 1 ? "PyMuPDF Extraction Complete" : "Evaluating Corporate & Checklist Files"}
                  </p>
                </div>
              </div>

              {/* Store MongoDB Step */}
              <div className={`flex gap-3 items-start ${stage < 2 ? "opacity-45" : ""}`}>
                <div className={`mt-0.5 w-5 h-5 rounded-full flex items-center justify-center text-white text-[10px] shrink-0 ${
                  stage > 2 ? "bg-green-500" : (stage === 2 && currentStepIndex === 1) ? "bg-indigo-600 animate-pulse" : (stage === 2 && currentStepIndex > 1) ? "bg-green-500" : "bg-slate-300"
                }`}>
                  {(stage > 2 || (stage === 2 && currentStepIndex > 1)) ? "✓" : "2"}
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-800">Vector Embedding Store [768d] — Atlas</p>
                  <p className="text-[10px] text-slate-400 leading-snug">
                    {stage > 2 || (stage === 2 && currentStepIndex > 1) ? "Completed Atlas ingestion" : (stage === 2 && currentStepIndex === 1) ? "Chunking and calculating distances..." : "Compliance mapping indexing"}
                  </p>
                </div>
              </div>

              {/* Detect Gaps Step */}
              <div className={`flex gap-3 items-start ${stage < 2 ? "opacity-45" : ""}`}>
                <div className={`mt-0.5 w-5 h-5 rounded-full flex items-center justify-center text-white text-[10px] shrink-0 ${
                  stage > 2 ? "bg-green-500" : (stage === 2 && currentStepIndex === 2) ? "bg-indigo-600 animate-pulse" : (stage === 2 && currentStepIndex > 2) ? "bg-green-500" : "bg-slate-300"
                }`}>
                  {(stage > 2 || (stage === 2 && currentStepIndex > 2)) ? "✓" : "3"}
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-800">Similarity Scan via Atlas Vector Search &lt; 0.75</p>
                  <p className="text-[10px] text-slate-400 leading-snug">
                    {stage > 2 || (stage === 2 && currentStepIndex > 2) ? "All compliance vector gaps mapped" : (stage === 2 && currentStepIndex === 2) ? "Analyzing similarity vectors..." : "Cosine threshold checks"}
                  </p>
                </div>
              </div>

              {/* Human Checkpoint Step */}
              <div className={`flex gap-3 items-start ${stage < 3 ? "opacity-45" : ""}`}>
                <div className={`mt-0.5 w-5 h-5 rounded-full flex items-center justify-center text-white text-[10px] shrink-0 ${
                  stage === 4 ? "bg-green-500" : stage === 3 ? "bg-orange-500 ring-2 ring-orange-100 animate-pulse" : "bg-slate-300"
                }`}>
                  {stage === 4 ? "✓" : "!"}
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-900 font-bold">Human Checkpoint</p>
                  <p className="text-[10px] text-orange-600 font-medium leading-snug">
                    {stage === 4 ? "Approval Decisions Finalized" : stage === 3 ? "Awaiting Approval [High Risk]" : "Verification checkpoint step"}
                  </p>
                </div>
              </div>

              {/* Final Report Step */}
              <div className={`flex gap-3 items-start ${stage < 4 ? "opacity-45" : ""}`}>
                <div className={`mt-0.5 w-5 h-5 rounded-full bg-slate-300 flex items-center justify-center text-white text-[10px] shrink-0 ${
                  stage === 4 ? "bg-green-500" : ""
                }`}>
                  {stage === 4 ? "✓" : "5"}
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-800">Final Report</p>
                  <p className="text-[10px] text-slate-400 leading-snug">
                    {stage === 4 ? "Report Generated Successfully" : "Pending Decisions..."}
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Token Usage Sidebar Section */}
          <div className="mt-auto p-4 bg-slate-50 border-t border-slate-200">
            <div className="flex justify-between text-xs mb-2">
              <span className="text-slate-500">Token Usage</span>
              <span className="text-slate-800 font-mono italic text-[10px]">
                {stage === 1 ? "0 (0.00 USD)" : stage === 2 ? "4,500 (~0.01 USD)" : "12,482 (~0.02 USD)"}
              </span>
            </div>
            <div className="w-full bg-slate-200 h-1 rounded-full overflow-hidden">
              <div className={`bg-blue-600 h-full transition-all duration-500 ${
                stage === 1 ? "w-0" : stage === 2 ? "w-1/4" : stage === 3 ? "w-3/4" : "w-full"
              }`}></div>
            </div>
          </div>
        </aside>

        {/* Right side contents panel */}
        <main className="flex-1 flex flex-col bg-white overflow-hidden">
          
          {/* Scrollable Container */}
          <div className="flex-1 overflow-y-auto p-4 md:p-6 lg:p-8 space-y-6">
            
            {/* Error Notification Alert */}
            {error && (
              <div className="p-3 bg-red-50 border-l-4 border-l-red-600 border border-red-200 rounded-lg flex items-start gap-2.5">
                <AlertOctagon className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-red-800 text-xs">Processing Failure</h4>
                  <p className="text-[11px] text-red-700 mt-0.5 leading-relaxed">{error}</p>
                </div>
              </div>
            )}

            {/* DB Connected Success Banner */}
            {dbConnectedMessage && !error && (
              <div className="p-3 bg-emerald-50 border-l-4 border-l-emerald-600 border border-emerald-200 rounded-lg flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-emerald-800 text-xs">System Connected</h4>
                  <p className="text-[11px] text-emerald-700 mt-0.5 leading-relaxed">{dbConnectedMessage}</p>
                </div>
              </div>
            )}

            {/* STAGE 1: PDF Selection & Upload */}
            {stage === 1 && (
              <div className="space-y-6 animate-fade-in max-w-5xl">
                {/* Introduction Header block info */}
                <div className="flex flex-col">
                  <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Enterprise Multi-Agent Auditor</h2>
                  <p className="text-sm text-slate-500">Coordinate isolated LLM security sub-agents to parse standard compliance assets in real-time.</p>
                </div>

                {/* Introduction Card */}
                <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
                  <h2 className="text-base font-semibold text-slate-900 flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-[#1F3A5F]" /> Real-Time Regulatory Assessment Loop
                  </h2>
                  <p className="text-xs text-slate-600 leading-relaxed">
                    AuditPilot inspects policy matrices against standard compliance structures to automatically vector corporate controls and isolate performance gaps. Uses gemini-1.5-flash text-to-embedding workaround (768 dimensions) with dual-agent calculations to measure exact cosine similarity thresholds (&lt; 0.75 similarity targets).
                  </p>
                  
                  <div className="p-2.5 bg-indigo-50/50 border border-indigo-100 rounded-lg flex items-center justify-between text-xs text-slate-600">
                    <span className="flex items-center gap-1.5 font-medium">
                      <Sparkles className="w-3.5 h-3.5 text-indigo-500 shrink-0" /> Fast-track evaluation with sample PDF templates
                    </span>
                    <button 
                      onClick={loadSamples}
                      className="px-2.5 py-1 bg-white border border-indigo-200 text-indigo-600 hover:bg-indigo-50 rounded text-xs font-semibold shadow-xs transition-colors cursor-pointer"
                    >
                      Load Sample Policy &amp; Checklist
                    </button>
                  </div>
                </div>

                {/* Upload Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* PDF 1: Corporate Policy */}
                  <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col">
                    <div className="flex items-center justify-between mb-3 border-b border-slate-100 pb-2">
                      <h3 className="font-semibold text-slate-800 text-xs uppercase tracking-wide flex items-center gap-1.5">
                        <FileCheck className="w-4 h-4 text-[#1F3A5F]" /> Corporate Policy
                      </h3>
                      <span className="text-[9px] bg-slate-100 text-slate-500 px-2 py-0.5 rounded font-mono">Policy PDF</span>
                    </div>
                    
                    <div 
                      onDragOver={(e) => handleDragOver(e, "policy")}
                      onDragLeave={() => handleDragLeave("policy")}
                      onDrop={(e) => handleDrop(e, "policy")}
                      onClick={() => policyInputRef.current?.click()}
                      className={`flex-1 min-h-[140px] border-2 border-dashed rounded-lg flex flex-col items-center justify-center p-4 text-center cursor-pointer transition-all ${
                        isDraggingPolicy 
                          ? "border-[#1F3A5F] bg-[#1F3A5F]/5" 
                          : policyFile 
                            ? "border-emerald-300 bg-emerald-50/20" 
                            : "border-slate-200 hover:border-slate-300 bg-slate-50/50"
                      }`}
                    >
                      <input 
                        type="file" 
                        ref={policyInputRef}
                        className="hidden" 
                        accept=".pdf"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) {
                            setPolicyFile(file);
                            setSampleMode(false);
                          }
                        }}
                      />
                      
                      {policyFile ? (
                        <div className="animate-fade-in">
                          <div className="bg-emerald-100 p-2.5 rounded-full text-emerald-600 inline-block mb-1.5">
                            <FileText className="w-5 h-5" />
                          </div>
                          <p className="text-xs font-semibold text-slate-700 max-w-[200px] truncate mx-auto">{policyFile.name}</p>
                          <p className="text-[10px] text-slate-500">{(policyFile.size / 1024).toFixed(1)} KB • PDF Document</p>
                          <span className="inline-block mt-2 text-[9px] bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded font-medium">Selected</span>
                        </div>
                      ) : (
                        <div>
                          <UploadCloud className="w-6 h-6 text-slate-400 mb-1.5 mx-auto" />
                          <p className="text-xs font-medium text-slate-700">Company Security Policy</p>
                          <p className="text-[10px] text-slate-500 mt-0.5">Drag &amp; drop policy PDF here or browse</p>
                        </div>
                      )}
                    </div>
                    {policyFile && <PdfPreview file={policyFile} type="policy" />}
                  </div>

                  {/* PDF 2: Checklist */}
                  <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col">
                    <div className="flex items-center justify-between mb-3 border-b border-slate-100 pb-2">
                      <h3 className="font-semibold text-slate-800 text-xs uppercase tracking-wide flex items-center gap-1.5">
                        <ShieldCheck className="w-4 h-4 text-[#1F3A5F]" /> Compliance Checklist
                      </h3>
                      <span className="text-[9px] bg-slate-100 text-slate-500 px-2 py-0.5 rounded font-mono">ISO 27001 Checklist</span>
                    </div>
                    
                    <div 
                      onDragOver={(e) => handleDragOver(e, "checklist")}
                      onDragLeave={() => handleDragLeave("checklist")}
                      onDrop={(e) => handleDrop(e, "checklist")}
                      onClick={() => checklistInputRef.current?.click()}
                      className={`flex-1 min-h-[140px] border-2 border-dashed rounded-lg flex flex-col items-center justify-center p-4 text-center cursor-pointer transition-all ${
                        isDraggingChecklist 
                          ? "border-[#1F3A5F] bg-[#1F3A5F]/5" 
                          : checklistFile 
                            ? "border-emerald-300 bg-emerald-50/20" 
                            : "border-slate-200 hover:border-slate-300 bg-slate-50/50"
                      }`}
                    >
                      <input 
                        type="file" 
                        ref={checklistInputRef}
                        className="hidden" 
                        accept=".pdf"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) {
                            setChecklistFile(file);
                            setSampleMode(false);
                          }
                        }}
                      />
                      
                      {checklistFile ? (
                        <div className="animate-fade-in">
                          <div className="bg-emerald-100 p-2.5 rounded-full text-emerald-600 inline-block mb-1.5">
                            <FileText className="w-5 h-5" />
                          </div>
                          <p className="text-xs font-semibold text-slate-700 max-w-[200px] truncate mx-auto">{checklistFile.name}</p>
                          <p className="text-[10px] text-slate-500">{(checklistFile.size / 1024).toFixed(1)} KB • PDF Document</p>
                          <span className="inline-block mt-2 text-[9px] bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded font-medium">Selected</span>
                        </div>
                      ) : (
                        <div>
                          <UploadCloud className="w-6 h-6 text-slate-400 mb-1.5 mx-auto" />
                          <p className="text-xs font-medium text-slate-700">ISO 27001 Checklist</p>
                          <p className="text-[10px] text-slate-500 mt-0.5">Drag &amp; drop checklist PDF here or browse</p>
                        </div>
                      )}
                    </div>
                    {checklistFile && <PdfPreview file={checklistFile} type="checklist" />}
                  </div>
                </div>

                {/* Submit CTA */}
                <button
                  onClick={runAudit}
                  disabled={!policyFile || !checklistFile}
                  className={`w-full py-3 px-5 rounded-lg font-bold shadow text-xs uppercase tracking-wider flex items-center justify-center gap-2 cursor-pointer transition-all ${
                    policyFile && checklistFile
                      ? "bg-[#1F3A5F] hover:bg-[#152943] text-white"
                      : "bg-slate-200 text-slate-400 cursor-not-allowed"
                  }`}
                >
                  Run Compliance Audit Agent <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* STAGE 2: Processing Flow */}
            {stage === 2 && (
              <div className="max-w-xl mx-auto py-8 text-center space-y-6 animate-fade-in">
                <div className="inline-block relative">
                  <div className="w-16 h-16 border-4 border-slate-100 border-t-[#1F3A5F] rounded-full animate-spin"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Sparkles className="w-6 h-6 text-indigo-500 animate-pulse" />
                  </div>
                </div>

                <div className="space-y-1">
                  <h2 className="text-lg font-semibold text-slate-900">AuditPilot Agent Executing</h2>
                  <p className="text-[11px] text-slate-500 font-mono">Running vector matching model pipelines...</p>
                </div>

                {/* Processing list */}
                <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm space-y-3.5 text-left">
                  {pipelineSteps.map((step, idx) => {
                    const isActive = idx === currentStepIndex;
                    const isCompleted = idx < currentStepIndex;
                    
                    return (
                      <div key={idx} className="flex gap-3 items-center">
                        <div className="shrink-0 animate-fade-in">
                          {isCompleted ? (
                            <div className="w-5 h-5 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center">
                              <Check className="w-3 h-3" />
                            </div>
                          ) : isActive ? (
                            <div className="w-5 h-5 rounded-full bg-indigo-50 border border-indigo-500 text-indigo-600 flex items-center justify-center animate-pulse flex-row">
                              <div className="w-1.5 h-1.5 bg-indigo-600 rounded-full"></div>
                            </div>
                          ) : (
                            <div className="w-5 h-5 rounded-full bg-slate-100 border border-slate-200 text-slate-400 flex items-center justify-center text-[9px] font-mono">
                              {idx + 1}
                            </div>
                          )}
                        </div>
                        
                        <div>
                          <h4 className={`text-xs font-semibold ${isActive ? "text-indigo-600" : isCompleted ? "text-slate-800" : "text-slate-400"}`}>
                            {step.label}
                          </h4>
                          <p className="text-[10px] text-slate-400 leading-tight">{step.desc}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>

                <p className="text-[11px] text-slate-400 font-medium">Please wait. Parsing layers, projecting metrics coordinates and discovering gaps...</p>
              </div>
            )}

            {/* STAGE 3: Human Verification Checkpoint */}
            {stage === 3 && (
              <div className="space-y-6 animate-fade-in">
                {/* Header Counters exactly matching layout arrangement of Design HTML */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 pb-4">
                  <div className="flex flex-col">
                    <h3 className="text-xl font-bold text-slate-900 tracking-tight">Agent Paused — Human Approval Required</h3>
                    <p className="text-slate-500 text-xs">
                      The compliance auditor must review all <span className="font-semibold text-red-600 uppercase">{highRiskGaps.length} findings</span> to proceed with audit report generation.
                    </p>
                  </div>

                  <div className="flex gap-2.5">
                    <div className="bg-red-50 border border-red-250 px-3 py-1.5 rounded-lg text-center min-w-[90px]">
                      <p className="text-[9px] text-red-650 font-bold uppercase tracking-wider">High/Critical</p>
                      <p className="text-lg font-black text-red-700 leading-none">
                        {highRiskGaps.filter(g => g.risk_level === "High" || g.risk_level === "Critical").length}
                      </p>
                    </div>
                    <div className="bg-orange-50 border border-orange-200 px-3 py-1.5 rounded-lg text-center min-w-[70px]">
                      <p className="text-[9px] text-orange-655 font-bold uppercase tracking-wider">Med/Low</p>
                      <p className="text-lg font-black text-orange-700 leading-none">
                        {highRiskGaps.filter(g => g.risk_level === "Medium" || g.risk_level === "Low").length}
                      </p>
                    </div>
                    <div className="bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg text-center min-w-[85px]">
                      <p className="text-[9px] text-slate-655 font-bold uppercase tracking-wider">Reviewed</p>
                      <p className="text-lg font-black text-slate-700 leading-none">
                        {Object.keys(decisions).length}/{highRiskGaps.length}
                      </p>
                    </div>
                  </div>
                </div>

                {/* List all gaps as flat solid cards matching aesthetic */}
                <div className="space-y-4">
                  {highRiskGaps.map((item) => {
                    const isEscalated = decisions[item.control_id] === "escalate";
                    const isAccepted = decisions[item.control_id] === "accept_risk";
                    const riskTheme = getRiskColor(item.risk_level);

                    return (
                      <div 
                        key={item.control_id}
                        className={`border-2 rounded-xl p-4 bg-white shadow-xs flex flex-col md:flex-row gap-4 transition-all ${
                          isEscalated 
                            ? "border-green-500 bg-green-50/10" 
                            : isAccepted 
                              ? "border-slate-400 bg-slate-50/10" 
                              : "border-slate-200"
                        }`}
                      >
                        <div className="flex-1 space-y-2.5">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="bg-slate-900 text-white font-mono px-2 py-0.5 rounded text-[10px] font-semibold shrink-0">
                              {item.control_id}
                            </span>
                            <span className="text-xs font-bold text-slate-800 uppercase tracking-tight">
                              {item.title || "Checklist Compliance Requirement"}
                            </span>
                            <span className={`text-[9px] px-2 py-0.5 font-bold rounded uppercase ${riskTheme.bg}`}>
                              {item.risk_level} RISK
                            </span>
                          </div>

                          <div className="text-xs text-slate-700 space-y-1.5">
                            <p className="leading-relaxed">
                              <span className="font-semibold text-slate-800 block text-[9px] uppercase tracking-wide opacity-75">Checklist Requirement</span>
                              {item.description}
                            </p>
                            
                            <p className="leading-relaxed bg-red-50/30 p-2 rounded border border-red-100/40">
                              <span className="font-semibold text-slate-800 block text-[9px] uppercase tracking-wide opacity-75">Gap Detected / Reasoning</span>
                              {item.gap_reason}
                            </p>

                            {item.evidence && item.evidence !== "None" && (
                              <p className="leading-relaxed bg-slate-50 p-2 rounded border border-slate-200/50">
                                <span className="font-semibold text-slate-800 block text-[9px] uppercase tracking-wide opacity-75">Matched Policy Evidence</span>
                                <code className="text-[10px] font-sans italic text-slate-600 block pt-0.5 select-text">"{item.evidence}"</code>
                              </p>
                            )}

                            {item.recommended_action && (
                              <p className="leading-relaxed bg-indigo-50/30 p-2 rounded border border-indigo-100/50">
                                <span className="font-semibold text-slate-800 block text-[9px] uppercase tracking-wide opacity-75">Remediation Action Advice</span>
                                {item.recommended_action}
                              </p>
                            )}
                          </div>

                          <div className="flex gap-6 text-[10px] pt-1.5 border-t border-slate-100">
                            <div className="flex flex-col">
                              <span className="text-slate-400 font-bold uppercase text-[9px]">Suggested Owner</span>
                              <span className="text-indigo-650 font-semibold">{item.owner}</span>
                            </div>
                            <div className="flex flex-col">
                              <span className="text-slate-400 font-bold uppercase text-[9px]">SLA Priority Target</span>
                              <span className="text-slate-700 font-semibold">
                                {item.risk_level === "Critical" ? "Immediate (24 hours)" : item.risk_level === "High" ? "SLA 15 Days" : "Standard (30 Days)"}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Right Buttons column matches exact look & feel of High Density theme markup */}
                        <div className="flex flex-col gap-2 w-56 shrink-0 justify-center">
                          <button
                            onClick={() => selectDecision(item.control_id, "escalate")}
                            className={`w-full py-2 px-3 rounded text-xs font-bold border-2 transition-all cursor-pointer truncate ${
                              isEscalated 
                                ? "bg-green-600 border-green-600 text-white shadow-xs" 
                                : "border-green-600 text-green-600 hover:bg-green-600 hover:text-white"
                            }`}
                          >
                            ESCALATE
                          </button>
                          <button
                            onClick={() => selectDecision(item.control_id, "accept_risk")}
                            className={`w-full py-2 px-3 rounded text-xs font-bold border-2 transition-all cursor-pointer truncate ${
                              isAccepted 
                                ? "bg-slate-700 border-slate-700 text-white shadow-xs" 
                                : "border-slate-200 text-slate-400 hover:text-slate-500 hover:border-slate-350"
                            }`}
                          >
                            ACCEPT RISK
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Actions bottom banner */}
                <div className="h-18 bg-slate-50 border border-slate-200 p-4 rounded-xl flex items-center justify-between shrink-0">
                  <div className="flex items-center gap-2">
                    <span className={`w-2.5 h-2.5 rounded-full ${Object.keys(decisions).length === highRiskGaps.length ? "bg-green-500" : "bg-red-500 animate-pulse"}`}></span>
                    <p className="text-xs text-slate-650 font-semibold">
                      <span>Review Status:</span> {Object.keys(decisions).length} of {highRiskGaps.length} findings reviewed
                    </p>
                  </div>
                  <button
                    onClick={submitDecisions}
                    disabled={isSubmittingDecisions || Object.keys(decisions).length < highRiskGaps.length}
                    className={`px-6 py-2 rounded-lg font-bold text-xs uppercase tracking-widest transition-all shadow-sm flex items-center gap-2 ${
                      Object.keys(decisions).length === highRiskGaps.length && !isSubmittingDecisions
                        ? "bg-[#1F3A5F] hover:bg-[#152943] text-white cursor-pointer"
                        : "bg-slate-200 text-slate-400 cursor-not-allowed"
                    }`}
                  >
                    {isSubmittingDecisions ? "Generating Alignment Report..." : "Generate Final Audit Report"}
                  </button>
                </div>
              </div>
            )}

            {/* STAGE 4: Audit Synthesis Report Screen */}
            {stage === 4 && (
              <div className="space-y-6 animate-fade-in">
                {/* Header Counters */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 pb-4">
                  <div className="flex flex-col">
                    <h3 className="text-xl font-bold text-slate-900 tracking-tight">Audit Posture Synthesis Finished</h3>
                    <p className="text-slate-500 text-xs">Standardized compliance analysis generated successfully.</p>
                  </div>

                  <div className="flex gap-2.5">
                    <div className="bg-[#1F3A5F] text-white px-3 py-1.5 rounded-lg text-center min-w-[100px]">
                      <p className="text-[9px] text-blue-205 font-bold uppercase tracking-wider">Overall Posture</p>
                      <p className="text-xs font-bold leading-normal truncate max-w-[120px]">{getGradeBadge(gaps).text}</p>
                    </div>
                    <div className="bg-orange-50 border border-orange-200 px-3 py-1.5 rounded-lg text-center min-w-[70px]">
                      <p className="text-[9px] text-orange-650 font-bold uppercase tracking-wider">Identified Gaps</p>
                      <p className="text-lg font-black text-orange-700 leading-none">{gaps.length}</p>
                    </div>
                  </div>
                </div>

                {/* Thread ID Token bar */}
                <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 flex flex-wrap items-center justify-between gap-3 text-xs">
                  <div>
                    <span className="text-slate-400 font-bold uppercase text-[9px] block">Audit Session Token</span>
                    <span className="font-mono text-slate-800 font-semibold text-[11px] select-all">{sessionId}</span>
                  </div>
                  <div className="flex gap-2">
                    <button 
                      onClick={copyReportJson}
                      className="p-1.5 bg-[#1F3A5F]/5 hover:bg-[#1F3A5F]/15 text-[#1F3A5F] rounded transition-colors cursor-pointer text-xs flex items-center gap-1.5 font-semibold"
                    >
                      {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>Copy JSON Payload</span>
                    </button>
                    <button 
                      onClick={downloadReportHtml}
                      className="px-3 py-1.5 bg-[#1F3A5F] hover:bg-[#152943] text-white rounded transition-all cursor-pointer text-xs flex items-center gap-1.5 font-bold shadow-xs active:scale-95"
                    >
                      <FileText className="w-3.5 h-3.5" />
                      <span>Download Audit Report (HTML)</span>
                    </button>
                  </div>
                </div>

                {/* Double Panel Layout */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                  {/* Executive summary markdown print */}
                  <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
                    <div className="flex items-center justify-between border-b border-slate-100 pb-2">
                      <h3 className="font-semibold text-sm text-slate-900 flex items-center gap-1.5">
                        <FileText className="w-4 h-4 text-[#1F3A5F]" /> Audit Executive Findings
                      </h3>
                      <span className="text-[9px] font-mono text-emerald-700 font-bold uppercase bg-emerald-50 px-2 py-0.5 rounded border border-emerald-100">
                        Verified
                      </span>
                    </div>

                    <div className="text-xs text-slate-705 space-y-3 leading-relaxed">
                      {finalReport ? (
                        <SimpleMarkdown text={finalReport} />
                      ) : (
                        <p className="text-slate-500 italic">No report context generated.</p>
                      )}
                    </div>
                  </div>

                  {/* Findings Categorization sidebar listing */}
                  <div className="lg:col-span-5 space-y-3">
                    <h4 className="text-xs font-semibold text-slate-800 uppercase tracking-widest">Indexed Compliance Deficiencies</h4>
                    
                    <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
                      {gaps.map((gap, index) => {
                        const color = getRiskColor(gap.risk_level);
                        return (
                          <div 
                            key={index}
                            className={`bg-white border border-slate-200 rounded-xl p-3.5 shadow-xs space-y-2.5 ${color.card}`}
                          >
                            <div className="flex justify-between items-start gap-1">
                              <div className="flex items-center gap-1.5">
                                <span className="text-[9px] font-mono bg-slate-100 text-slate-700 px-1.5 py-0.5 rounded shrink-0 font-semibold border border-slate-150">
                                  {gap.control_id}
                                </span>
                                <h4 className="text-xs font-bold text-slate-900 leading-tight">{gap.description}</h4>
                              </div>
                              <span className={`text-[9px] px-1.5 py-0.5 font-semibold rounded ${color.bg} shrink-0`}>
                                {gap.risk_level}
                              </span>
                            </div>

                            <p className="text-[11px] text-slate-600 leading-snug bg-slate-50 p-2 rounded">
                              <span className="font-semibold text-slate-800 block text-[9px] uppercase tracking-wide opacity-75">Finding Issue</span>
                              {gap.gap_reason}
                            </p>

                            <p className="text-[11px] text-slate-600 leading-snug bg-slate-50 p-2 rounded">
                              <span className="font-semibold text-slate-800 block text-[9px] uppercase tracking-wide opacity-75">Audit Action Advice</span>
                              {gap.recommended_action}
                            </p>

                            <div className="flex justify-between items-center text-[10px] border-t border-slate-100 pt-2 shrink-0">
                              <span className="text-slate-500 font-semibold">Assigned Route: <strong className="text-indigo-650">{gap.owner}</strong></span>
                              {gap.human_decision && (
                                <span className="bg-slate-105 text-slate-800 border border-slate-200 px-2 py-0.5 rounded text-[9px] font-mono font-bold uppercase shrink-0">
                                  {gap.human_decision === "escalate" ? "Escalated" : "Risk Accepted"}
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            )}

          </div>

          {/* Fixed Footer */}
          <footer className="h-12 bg-white border-t border-slate-200 px-6 flex items-center justify-between text-xs text-slate-400 shrink-0 select-none">
            <p>© 2026 AuditPilot Systems Inc.</p>
            <div className="flex gap-4 font-medium text-slate-500">
              <span className="hover:text-[#1F3A5F] cursor-pointer">ISO Frameworks</span>
              <span>•</span>
              <span className="hover:text-[#1F3A5F] cursor-pointer">Security Hub</span>
            </div>
          </footer>

        </main>
      </div>
    </div>

    {/* Printable PDF Report Panel */}
    <div className="hidden print:block max-w-4xl mx-auto p-8 bg-white text-slate-950 font-sans text-xs leading-normal select-text">
      {/* Letterhead */}
      <div className="flex justify-between items-center border-b-2 border-[#1F3A5F] pb-4 mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-[#1F3A5F] rounded-md flex items-center justify-center font-bold text-white text-base">
            AP
          </div>
          <div>
            <h1 className="text-xl font-bold text-[#1F3A5F] tracking-tight">AuditPilot Report Matrix</h1>
            <p className="text-[10px] text-slate-500 font-medium">Automated Enterprise Compliance Assessment Platform</p>
          </div>
        </div>
        <div className="text-right text-[10px] space-y-0.5">
          <p className="font-bold text-slate-900">CLASS: FORMAL REASSESSMENT</p>
          <p className="font-mono text-slate-500">Session ID: {sessionId || "N/A"}</p>
          <p className="font-mono text-slate-500">Audit Date: {new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}</p>
        </div>
      </div>

      {/* Metadata Metrics Row */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="border border-slate-205 rounded-lg p-3 bg-slate-50 flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider block">Security Posture Status</span>
          <span className={`text-xs font-extrabold mt-1 uppercase ${getGradeBadge(gaps).colorClass}`}>
            {getGradeBadge(gaps).text}
          </span>
        </div>
        <div className="border border-slate-205 rounded-lg p-3 bg-slate-50 flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider block">Indexed Deficiencies</span>
          <span className="text-xs font-black mt-1 text-slate-900 font-mono">
            {gaps.length} Compliance Gaps
          </span>
        </div>
        <div className="border border-slate-205 rounded-lg p-3 bg-slate-50 flex flex-col justify-center">
          <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider block">Regulatory Frameworks</span>
          <span className="text-xs font-semibold mt-1 text-slate-800">
            ISO / IEC 27001 Standards
          </span>
        </div>
      </div>

      {/* Gaps Summary Category Table */}
      <div className="mb-6">
        <h3 className="font-bold text-[#1F3A5F] text-[11px] uppercase tracking-wider mb-2">Findings Metrics Dashboard</h3>
        <table className="min-w-full divide-y divide-slate-200 border border-slate-200 rounded-lg overflow-hidden text-[10px]">
          <thead className="bg-slate-100 font-bold text-slate-700">
            <tr>
              <th className="px-4 py-1.5 text-left">Category Field</th>
              <th className="px-4 py-1.5 text-center">Identified Count</th>
              <th className="px-4 py-1.5 text-left">Target Remediation Impact</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 bg-white text-slate-800">
            <tr>
              <td className="px-4 py-1.5 font-medium">Total Compliance Gaps</td>
              <td className="px-4 py-1.5 text-center font-bold font-mono">{gaps.length}</td>
              <td className="text-slate-500 px-4 py-1.5">Unaddressed elements needing system configuration updates.</td>
            </tr>
            <tr>
              <td className="px-4 py-1.5 font-medium text-red-700">High Risk Gaps</td>
              <td className="px-4 py-1.5 text-center font-bold font-mono text-red-700">{gaps.filter(g => g.risk_level === "High").length}</td>
              <td className="text-red-500 px-4 py-1.5 font-medium">Critical non-compliance issues flagging urgent human escalations.</td>
            </tr>
            <tr>
              <td className="px-4 py-1.5 font-medium text-orange-700">Medium Risk Gaps</td>
              <td className="px-4 py-1.5 text-center font-bold font-mono text-orange-700">{gaps.filter(g => g.risk_level === "Medium").length}</td>
              <td className="text-orange-500 px-4 py-1.5 font-medium">Operational security gaps or configuration irregularities.</td>
            </tr>
            <tr>
              <td className="px-4 py-1.5 font-medium text-yellow-700">Low Risk Gaps</td>
              <td className="px-4 py-1.5 text-center font-bold font-mono text-yellow-700">{gaps.filter(g => g.risk_level === "Low").length}</td>
              <td className="text-yellow-500 px-4 py-1.5 font-medium">Minor policy recommendations or standard operational optimizations.</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Full Findings Section Grid */}
      <div className="mb-6 print-page-break">
        <h3 className="font-bold text-[#1F3A5F] text-[11px] uppercase tracking-wider mb-2">Compliance Gaps Indexing Matrix</h3>
        <div className="space-y-3">
          {gaps.length > 0 ? (
            gaps.map((gap, index) => (
              <div key={index} className="border border-slate-300 rounded-lg p-3 bg-white space-y-2">
                <div className="flex justify-between items-center border-b border-slate-150 pb-1.5">
                  <div className="flex items-center gap-2">
                    <span className="font-mono bg-slate-900 text-white px-2 py-0.5 rounded text-[8px] font-bold">
                      {gap.control_id}
                    </span>
                    <span className="font-bold text-slate-900 text-[10px]">
                      {gap.description}
                    </span>
                  </div>
                  <span className="text-[8px] px-2 py-0.5 font-bold rounded bg-slate-200 text-slate-700 uppercase">
                    {gap.risk_level} RISK
                  </span>
                </div>
                <p className="text-[10px] text-slate-800">
                  <strong className="text-slate-950 font-semibold block text-[8px] uppercase tracking-wide opacity-80">Finding description / gap reason:</strong> 
                  {gap.gap_reason}
                </p>
                <p className="text-[10px] text-slate-800">
                  <strong className="text-slate-950 font-semibold block text-[8px] uppercase tracking-wide opacity-80">Recommended remediation action:</strong> 
                  {gap.recommended_action}
                </p>
                <div className="flex justify-between items-center text-[8px] text-slate-500 pt-1.5 border-t border-slate-100">
                  <span>Department Lead Assigned: <strong>{gap.owner}</strong></span>
                  {gap.human_decision && (
                    <span className="bg-slate-100 text-slate-800 font-mono font-bold uppercase rounded px-2 py-0.5">
                      Vetting decision: {gap.human_decision === "approved" ? "Escalated to Team" : "Accepted business risk"}
                    </span>
                  )}
                </div>
              </div>
            ))
          ) : (
            <p className="text-slate-450 italic">No formal deficiency issues captured in active log.</p>
          )}
        </div>
      </div>

      {/* Synthesis Executive Report */}
      <div className="space-y-4">
        <h3 className="font-bold text-[#1F3A5F] text-[11px] uppercase tracking-wider mb-2 border-b border-[#1F3A5F] pb-1">
          ISO 27001 Auditor Executive Synthesis
        </h3>
        {finalReport ? (
          <SimpleMarkdown text={finalReport} />
        ) : (
          <p className="text-slate-500 italic">No report context compiled.</p>
        )}
      </div>

      {/* Print Footer */}
      <div className="mt-8 border-t border-slate-200 pt-4 text-center text-[9px] text-slate-400 select-none">
        <p>This document was formally compiled by AuditPilot using real-time vector similarity scanners and Gemini models on {new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}.</p>
        <p>Audit Session Token: {sessionId} • Generated on Google AI Studio Cloud Container Architecture</p>
      </div>
    </div>
  </>
);
}
