# --- Build stage ---
# eclipse-temurin ships official JDK images per Java version; we install Maven
# 3.9.16 explicitly from Apache's archive rather than relying on a
# maven:3.9.16-eclipse-temurin-25 tag, since that combined tag isn't
# guaranteed to exist yet for a JDK this new. Pin verified at build time.
FROM eclipse-temurin:25-jdk AS build
ARG MAVEN_VERSION=3.9.16
RUN apt-get update && apt-get install -y --no-install-recommends wget tar && \
    wget -q https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz && \
    tar -xzf apache-maven-${MAVEN_VERSION}-bin.tar.gz -C /opt && \
    rm apache-maven-${MAVEN_VERSION}-bin.tar.gz && \
    ln -s /opt/apache-maven-${MAVEN_VERSION}/bin/mvn /usr/local/bin/mvn && \
    apt-get purge -y wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pom.xml .
# Cache dependency resolution as its own layer
RUN mvn -q dependency:go-offline
COPY src ./src
RUN mvn -q clean package -DskipTests

# --- Run stage ---
FROM eclipse-temurin:25-jre
WORKDIR /app
COPY --from=build /app/target/*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
