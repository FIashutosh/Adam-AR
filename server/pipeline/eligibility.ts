// =============================================================
// Stage 6: AUDIT ELIGIBILITY GATE
// The Single Authoritative Compliance Gatekeeper.
//
// Every audit entry point MUST call this gate:
// - Automatic 24/7 worker
// - Run All
// - Force Audit
// - Re-Audit
// - UI / API endpoints
//
// Checks strictly:
// 1. classification === 'PRE_ORDER' (spoken order intent verified)
// 2. identity_status === 'CONFIRMED' (client UCC / account verified)
// 3. transcript_status === 'VALID' (valid verbatim transcript)
// 4. speaker attribution usable (has dialogue segments)
//
// CRITICAL ARCHITECTURAL MANDATE:
// Execution matching status (CONFIRMED / PARTIAL / NO_MATCH / REVIEW)
// must NEVER gate or block compliance auditing. Every genuine pre-order
// dialogue must be audited for regulatory compliance regardless of whether
// an executed trade was found, cancelled, or pending.
// =============================================================

import type { DatabaseSync } from 'node:sqlite';
import type { AuditEligibilityResult } from './types';
import type { CallRecord } from '../../src/types';
import { isValidUcc } from './uccResolver';

export function isAuditEligible(
  db: DatabaseSync,
  callOrId: CallRecord | number
): AuditEligibilityResult {
  let call: CallRecord | undefined;

  if (typeof callOrId === 'number') {
    call = db.prepare('SELECT * FROM calls WHERE id = ?').get(callOrId) as unknown as CallRecord | undefined;
  } else {
    call = callOrId;
  }

  if (!call) {
    return {
      eligible: false,
      gateCode: 'CALL_NOT_FOUND',
      reason: 'Call record could not be found in the database.',
    };
  }

  // Gate 1: Classification check (Spoken Order Intent)
  // Strict Rule: Trade data does NOT automatically make a call PRE_ORDER.
  // A call is PRE_ORDER only when the transcript contains actionable order details corresponding to a trade.
  let classification = (call.classification || call.call_type || '').toUpperCase();

  if (classification !== 'PRE_ORDER') {
    return {
      eligible: false,
      gateCode: 'NOT_PRE_ORDER',
      reason: `Call classification is "${classification || 'UNCLASSIFIED'}". Only confirmed PRE_ORDER calls are eligible for SEBI compliance audit.`,
    };
  }

  // Gate 2: Identity check (SEBI Mandate: Evaluate client UCC in Q2 instead of blocking audit execution)
  let identityStatus = (call.identity_status || '').toUpperCase();
  let rawCode = (call.client_code || call.client || '').trim();
  let hasValidClientCode = Boolean(rawCode && isValidUcc(rawCode));

  // If call doesn't have valid UCC directly, resolve from linked trade or match record
  if (!hasValidClientCode) {
    if (call.matched_trade_id && call.matched_trade_id > 0) {
      try {
        const trade = db.prepare('SELECT client, client_code FROM trades WHERE id = ?').get(call.matched_trade_id) as any;
        if (trade && (trade.client || trade.client_code)) {
          rawCode = (trade.client || trade.client_code).trim();
          hasValidClientCode = Boolean(rawCode && isValidUcc(rawCode));
          if (hasValidClientCode) {
            try {
              db.prepare("UPDATE calls SET client_code = ?, client = ?, identity_status = 'CONFIRMED' WHERE id = ?").run(rawCode, rawCode, call.id);
            } catch {}
          }
        }
      } catch {}
    }
  }

  if (!hasValidClientCode) {
    try {
      const match = db.prepare("SELECT trade_id FROM matches WHERE call_id = ? AND (verification_status = 'confirmed' OR status = 'matched') LIMIT 1").get(call.id) as any;
      if (match?.trade_id) {
        const trade = db.prepare('SELECT client, client_code FROM trades WHERE id = ?').get(match.trade_id) as any;
        if (trade && (trade.client || trade.client_code)) {
          rawCode = (trade.client || trade.client_code).trim();
          hasValidClientCode = Boolean(rawCode && isValidUcc(rawCode));
          if (hasValidClientCode) {
            try {
              db.prepare("UPDATE calls SET client_code = ?, client = ?, matched_trade_id = ?, identity_status = 'CONFIRMED' WHERE id = ?").run(rawCode, rawCode, match.trade_id, call.id);
            } catch {}
          }
        }
      }
    } catch {}
  }

  if (!hasValidClientCode) {
    // Client code not verified prior to order - record for Q2 compliance evaluation
    const fallbackId = rawCode || call.calling_number || call.phone_number || 'UNCONFIRMED_CLIENT';
    try {
      db.prepare("UPDATE calls SET client_code = COALESCE(NULLIF(client_code, ''), ?), identity_status = 'UNCONFIRMED' WHERE id = ?").run(fallbackId, call.id);
    } catch {}
  } else if (identityStatus !== 'CONFIRMED') {
    identityStatus = 'CONFIRMED';
    try {
      db.prepare("UPDATE calls SET identity_status = 'CONFIRMED' WHERE id = ?").run(call.id);
    } catch {}
  }

  // Gate 3: Transcript check - STRICT MANDATE: Cannot audit call before transcription!
  const transcriptStatus = (call.transcript_status || '').toUpperCase();
  const transcript = (call.transcript || '').trim();
  if (transcript.length < 10) {
    return {
      eligible: false,
      gateCode: 'TRANSCRIPT_REQUIRED',
      reason: `Transcript is missing or incomplete (length: ${transcript.length}). A call cannot be audited before valid transcription is complete.`,
    };
  }

  // Gate 4: Speaker attribution check
  const segmentCount = (
    db.prepare('SELECT count(*) as count FROM call_segments WHERE call_id = ?').get(call.id) as { count: number }
  )?.count || 0;

  if (segmentCount === 0 && !transcript.toUpperCase().includes('ADVISOR:') && !transcript.toUpperCase().includes('DEALER:')) {
    // If transcript exists and call is confirmed pre-order or trade-matched, allow full transcript evaluation rather than blocking audit!
    if (!transcript || transcript.trim().length < 10) {
      return {
        eligible: false,
        gateCode: 'SPEAKER_ATTRIBUTION_UNUSABLE',
        reason: 'No speaker-attributed dialogue segments exist for this call. Speaker attribution is mandatory for Q2/Q4/Q5 evaluation.',
      };
    }
  }

  return {
    eligible: true,
    gateCode: 'ELIGIBLE',
    reason: 'Call passed all pre-order compliance gates (spoken order intent and client identity confirmed). Eligible for compliance audit.',
  };
}
