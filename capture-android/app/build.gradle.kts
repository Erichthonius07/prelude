plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.prelude.capture"
    compileSdk = libs.versions.compileSdk.get().toInt()
    defaultConfig {
        applicationId = "com.prelude.capture"
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        versionCode = 1
        versionName = "0.1.0"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures { viewBinding = true }
}

// AGP 9.x built-in Kotlin: the Kotlin compiler is configured through this top-level
// extension, not android.kotlinOptions (removed in AGP 9) and not the standalone
// kotlin-android plugin.
kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

dependencies {
    implementation(project(":quantization-deploy"))
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.okhttp)

    testImplementation(libs.junit)
    // Real org.json on the JVM test classpath: the android.jar org.json stub throws in
    // unit tests, which would make the wire-payload tests unrunnable.
    testImplementation(libs.json)
}
