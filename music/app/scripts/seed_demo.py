from app.pipelines.daily_generation_pipeline import DailyGenerationPipeline

if __name__ == "__main__":
    print("Seeding demo tracks...")
    DailyGenerationPipeline.run()
    print("Done")
