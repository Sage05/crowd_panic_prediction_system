# Use the official Node.js image
FROM node:22

# Set the working directory
WORKDIR /app

# Copy package files first
COPY Frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy the frontend source
COPY Frontend/ .

# Expose the Vite development server port
EXPOSE 5173

# Start the frontend
CMD ["npm", "run", "dev", "--", "--host"]