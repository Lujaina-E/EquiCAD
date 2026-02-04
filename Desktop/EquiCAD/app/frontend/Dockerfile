FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

ENV PORT=3000

COPY . .
COPY .env .env.production


CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
