/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface DocumentChunk {
  document_name: string;
  document_type: "policy" | "checklist";
  chunk_index: number;
  content: string;
}

export interface PolicyControl {
  control_id: string;
  title: string;
  requirement_summary: string;
  source_text: string;
  category: string;
  confidence: number;
}

export interface ChecklistRequirement {
  requirement_id: string;
  title: string;
  required_control: string;
  category: string;
  severity_hint: string;
}

export interface Gap {
  control_id: string; // Mapped to requirement_id for compatibility
  title: string; // Requirement title
  description: string; // Full requirement or description
  gap_reason: string; // Compliance Verification reasoning
  evidence: string; // Matched evidence
  missing_requirement: string;
  recommended_action: string; // Remediation recommendation
  owner: "Security Team" | "IT Operations" | "Compliance Team" | "Data Protection Officer" | "Vendor Management Team";
  risk_level: "Critical" | "High" | "Medium" | "Low";
  confidence: number;
  human_decision?: "escalate" | "accept_risk";
}

export interface LangGraphState {
  session_id: string;
  policy_chunks: DocumentChunk[];
  checklist_chunks: DocumentChunk[];
  policy_controls: PolicyControl[];
  checklist_controls: ChecklistRequirement[];
  gaps: Gap[];
  awaiting_human: boolean;
  human_decisions: Record<string, "escalate" | "accept_risk">;
  final_report: string;
  status: "idle" | "parsing" | "analyzing" | "awaiting_approval" | "generating" | "complete" | "error";
  current_pipeline_step?: string;
  error: string | null;
}

export interface RetrievalMatch {
  requirement: ChecklistRequirement;
  matchedControl: PolicyControl | null;
  similarityScore: number;
}

export interface AuditGraphState extends LangGraphState {
  retrieval_matches: RetrievalMatch[];
  policy_buffer?: any; // Buffer type passed dynamically
  policy_name?: string;
  checklist_buffer?: any; // Buffer type passed dynamically
  checklist_name?: string;
  db?: any; // Database instance passed dynamically
}

export interface AuditSession {
  session_id: string;
  state: LangGraphState;
  updated_at: string;
}
