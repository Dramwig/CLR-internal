"""Persist controller transitions for future conversations (not notifications)."""
from datetime import datetime,timezone
from clr.utils.idea_a import write_json

def update(stage,completed,total,last_run,training_manifest='data/M4_v0/training_suite.json'):
    write_json('docs/M4_LIVE.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'completed_jobs':completed,'planned_jobs':total,'last_completed_run':str(last_run),'training_manifest':training_manifest,'evaluation_manifest':'data/M4_v0/evaluation_suite.json','final_review':'reports/M4_REVIEW.md','g1_approved':False})
