.PHONY: up down logs shell migrate test lint format seed-demo run-daily

up:
	docker-compose up -d --build

down:
	docker-compose down

logs:
	docker-compose logs -f

shell:
	docker-compose exec api bash

migrate:
	docker-compose exec api alembic upgrade head

test:
	docker-compose exec api pytest tests/

lint:
	docker-compose exec api ruff check app tests
	docker-compose exec api black --check app tests

format:
	docker-compose exec api ruff check --fix app tests
	docker-compose exec api black app tests

seed-demo:
	docker-compose exec api python -m app.scripts.seed_demo

run-daily:
	docker-compose exec api curl -X POST "http://localhost:8000/admin/run/daily-generate" -H "X-API-Key: $${INTERNAL_API_KEY:-dev_key}"
