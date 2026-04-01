from app.db import SessionLocal
from app.services.compilation_service import CompilationService
from app.utils.time import localnow


class WeeklyCompilationPipeline:
    @staticmethod
    def run():
        db = SessionLocal()
        try:
            comp_service = CompilationService(db)
            comp = comp_service.create_weekly_compilation(localnow().date())
            if comp:
                return str(comp.id)
            return None
        finally:
            db.close()
