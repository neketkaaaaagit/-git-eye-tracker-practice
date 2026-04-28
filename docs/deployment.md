# Развёртывание

## Вариант 1. Локальный запуск через Node.js

```bash
npm run dev
```

Приложение будет доступно на `http://localhost:8080`.

## Вариант 2. Локальный запуск через Python

```bash
python -m http.server 8080
```

## Вариант 3. Docker

### Сборка образа

```bash
docker build -t eye-tracker-stand .
```

### Запуск контейнера

```bash
docker run --rm -p 8080:80 eye-tracker-stand
```

## Вариант 4. Docker Compose

```bash
docker compose up --build
```

## Вариант 5. GitHub Pages

1. Запушить репозиторий на GitHub.
2. Убедиться, что основная ветка называется `main` или скорректировать `pages.yml`.
3. Включить GitHub Pages в настройках репозитория.
4. После push workflow `Deploy to GitHub Pages` соберёт и выложит `dist/`.

## Примечания по камере

- камера работает только в безопасном контексте: `localhost` или HTTPS;
- GitHub Pages работает по HTTPS, поэтому подходит для демонстрации;
- при открытии контейнерной версии через `http://localhost` камера также доступна.
