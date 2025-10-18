FROM node:20-alpine AS build-stage

WORKDIR /app

# Copiar solo package.json y package-lock.json para usar cache de npm install
COPY package*.json ./

# Instalar dependencias con legacy-peer-deps por compatibilidades
RUN npm install three
RUN npm i vue-router
RUN npm i -D @vitejs/plugin-vue
RUN npm install --legacy-peer-deps

# Copiar el resto del proyecto (incluyendo index.html, src/, vite.config.js)
COPY . .

# Build app
RUN npm run build

FROM nginx:stable-alpine AS production-stage

COPY --from=build-stage /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]