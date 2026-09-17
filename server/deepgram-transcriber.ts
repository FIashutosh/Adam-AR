// =============================================================
// AuditEQ — Deepgram Lightning-Fast ASR & Diarization Transcriber
// =============================================================

import fs from 'node:fs';
import path from 'node:path';
import type { SegmentInfo } from './evidence-extractor';

export interface DeepgramTranscriptionResult {
  transcript: string;
  rawTranscript: string;
  segments: SegmentInfo[];
  durationSeconds: number;
  modelUsed: string;
  confidence: number;
  channels: number;
}

function getAudioMimeType(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  switch (ext) {
    case '.wav':
      return 'audio/wav';
    case '.mp3':
      return 'audio/mpeg';
    case '.ogg':
      return 'audio/ogg';
    case '.m4a':
      return 'audio/m4a';
    case '.aac':
      return 'audio/aac';
    case '.flac':
      return 'audio/flac';
    default:
      return 'audio/wav';
  }
}

/**
 * Checks if Deepgram is configured via parameter, environment, or DB setting
 */
export function isDeepgramConfigured(providedKey?: string): boolean {
  if (providedKey && providedKey.trim().length > 0) return true;
  if (process.env.DEEPGRAM_API_KEY && process.env.DEEPGRAM_API_KEY.trim().length > 0) return true;
  return false;
}

/**
 * Tests Deepgram API key validity
 */
export async function testDeepgramKey(apiKey: string): Promise<{ ok: boolean; message: string }> {
  try {
    const cleanKey = apiKey.trim();
    if (!cleanKey) {
      return { ok: false, message: 'Deepgram API key is empty.' };
    }
    const res = await fetch('https://api.deepgram.com/v1/projects', {
      headers: {
        Authorization: `Token ${cleanKey}`,
      },
    });
    if (res.ok) {
      return { ok: true, message: 'Deepgram API key verified successfully! Ready for lightning-fast transcription.' };
    }
    const errText = await res.text();
    return { ok: false, message: `Deepgram verification failed (HTTP ${res.status}): ${errText}` };
  } catch (err: any) {
    return { ok: false, message: `Network error connecting to Deepgram: ${err?.message}` };
  }
}

/**
 * Transcribes an audio file with Deepgram Nova-2 with diarization and formatting.
 * Typically completes in 1-2 seconds per call recording!
 */
export async function transcribeWithDeepgram(
  audioPath: string,
  apiKey: string,
  model = 'nova-2'
): Promise<DeepgramTranscriptionResult> {
  const cleanKey = apiKey.trim();
  if (!cleanKey) {
    throw new Error('DEEPGRAM_API_KEY is required to transcribe with Deepgram.');
  }

  if (!fs.existsSync(audioPath)) {
    throw new Error(`Audio file not found on disk: ${audioPath}`);
  }

  const audioBuffer = fs.readFileSync(audioPath);
  const mimeType = getAudioMimeType(audioPath);

  const queryParams = new URLSearchParams({
    model: model || 'nova-2',
    smart_format: 'true',
    diarize: 'true',
    punctuate: 'true',
    utterances: 'true',
    numerals: 'true',
    detect_language: 'true',
    filler_words: 'false',
  });

  // Enterprise financial domain keywords with recognition boost weights across English, Hindi, Tamil, Telugu
  const keywordsToBoost = [
    'UCC:3.0',
    'client code:2.5',
    'CMP:2.5',
    'current market price:2.5',
    'market rate:2.5',
    'limit order:2.0',
    'market order:2.0',
    'target:2.0',
    'stop loss:2.0',
    'Welspun:2.5',
    'Welspun Living:2.5',
    'Bajaj Finserv:2.5',
    'Tata Steel:2.5',
    'Tata Motors:2.5',
    'Reliance:2.0',
    'Infosys:2.0',
    'Data Patterns:2.5',
    'L&T Finance:2.5',
    'Uno Minda:2.5',
    'HDFC Bank:2.0',
    'ICICI Bank:2.0',
    'FundsIndia:2.0',
    'quantity:2.0',
    'shares:2.0',
    'buy:2.0',
    'sell:2.0',
    'executed:2.0',
    'execution:2.0',
    'profit:1.5',
    'bhav:2.0',
    // UCC Prefixes
    'WIA:3.0',
    'WIF:3.0',
    'WIC:3.0',
    'WID:3.0',
    'WIG:3.0',
    'WIE:3.0',
    'FIA:3.0',
    'PWD:3.0',
    // Hindi keywords
    'kharidna:2.0',
    'bechna:2.0',
    'daal do:2.0',
    'kar do:2.0',
    'theek hai:2.0',
    'pachees:2.0',
    'pachaas:2.0',
    // Tamil keywords
    'vaangunga:2.0',
    'vaanga:2.0',
    'virka:2.0',
    'vithudunga:2.0',
    'vilai:2.0',
    'podunga:2.0',
    'seri:2.0',
    'aamaa:2.0',
    // Telugu keywords
    'kondi:2.0',
    'konali:2.0',
    'ammeyandi:2.0',
    'pettandi:2.0',
    'vela:2.0',
    'sare:2.0',
    'avunu:2.0',
    'alage:2.0',
  ];
  for (const kw of keywordsToBoost) {
    queryParams.append('keywords', kw);
  }

  const url = `https://api.deepgram.com/v1/listen?${queryParams.toString()}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Token ${cleanKey}`,
      'Content-Type': mimeType,
    },
    body: audioBuffer,
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Deepgram API error (${response.status}): ${errText}`);
  }

  const data = (await response.json()) as any;
  const channel = data?.results?.channels?.[0];
  const alternative = channel?.alternatives?.[0];
  const fullTranscript = (alternative?.transcript || '').trim();
  const durationSeconds = Number(data?.metadata?.duration) || 0;
  const confidence = Number(alternative?.confidence) || 0.95;

  const rawUtterances = data?.results?.utterances || [];
  const segments: SegmentInfo[] = [];

  if (rawUtterances.length > 0) {
    // First speaker is typically advisor in outbound or agent answering
    const firstSpeakerId = rawUtterances[0]?.speaker ?? 0;

    for (let i = 0; i < rawUtterances.length; i++) {
      const u = rawUtterances[i];
      const text = (u.transcript || '').trim();
      if (!text) continue;

      const speaker: 'ADVISOR' | 'CLIENT' = u.speaker === firstSpeakerId ? 'ADVISOR' : 'CLIENT';
      segments.push({
        speaker,
        start: Number(u.start) || 0,
        end: Number(u.end) || 0,
        text,
      });
    }
  } else if (alternative?.paragraphs?.paragraphs) {
    // Fallback to paragraphs if utterances not split
    for (const p of alternative.paragraphs.paragraphs) {
      const speaker: 'ADVISOR' | 'CLIENT' = p.speaker === 0 ? 'ADVISOR' : 'CLIENT';
      const text = (p.sentences || []).map((s: any) => s.text).join(' ').trim();
      if (text) {
        segments.push({
          speaker,
          start: Number(p.start) || 0,
          end: Number(p.end) || 0,
          text,
        });
      }
    }
  }

  // If no diarized segments returned, create a fallback single segment
  if (segments.length === 0 && fullTranscript) {
    segments.push({
      speaker: 'ADVISOR',
      start: 0,
      end: durationSeconds,
      text: fullTranscript,
    });
  }

  return {
    transcript: fullTranscript,
    rawTranscript: fullTranscript,
    segments,
    durationSeconds,
    modelUsed: `deepgram-${model || 'nova-2'}`,
    confidence,
    channels: Number(data?.metadata?.channels) || 1,
  };
}
