"""Developer tool: print a compact event trace for a demo script.  Usage: python scripts/trace_demo.py pune-demo"""
import json, sys
from kairos.config import load_config
from kairos.engine import EngineContext
from kairos.replay import run_script_sync
ctx = EngineContext(load_config())
evs = run_script_sync(ctx, sys.argv[1])
for e in evs:
    p = e.payload
    line = f"{e.timestamp:7.3f} {e.event_type:24s} u={e.utterance_id} sq={e.sub_query_id} trig={e.retrieval_trigger}"
    if e.event_type=="TranscriptChunkReceived": line += " '" + p['text'] + "'"
    if e.event_type=="RetrievalDecision":
        s=p.get("signals",{})
        line += f" DEC={p['decision']} {p['reasons']} s={s.get('s_hat')} th={s.get('theta')} rho={s.get('rho')} chi={s.get('chi')} k={s.get('kappa')}"
    if e.event_type=="TurnClassified": line += f" {p['turn_type']} {p['confidence']}"
    if e.event_type=="SubQueriesGenerated": line += " " + str([(a['candidate'],a['sub_query_id'],a['created'],a.get('footprint_jaccard')) for a in p['assignments']])
    if e.event_type=="RetrievalCompleted": line += " " + str(e.chunk_ids)
    if e.event_type=="EvidenceFused" and p['contradictions']: line += " CONTRA " + str([c['chunks'] for c in p['contradictions']])
    if e.event_type=="DeltaPlanComputed": line += " " + json.dumps({k:v['action'] for k,v in p['plan'].items()}) + str([a['sub_query_id'] for a in p['affected_sub_queries']])
    if e.event_type=="GroundingVerified": line += " " + json.dumps(p['summary'])
    if e.event_type=="TelemetryEmitted": line += " lead=" + str(p['retrieval_lead_s']) + " post_ms=" + str(p['post_utterance_latency_ms'])
    print(line)
    if e.event_type=="AnswerVersionCreated":
        print("   ---", p['kind'], json.dumps(p['diff']))
        print("   " + p['rendering'].replace("\n","\n   "))
