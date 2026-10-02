plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.prelude.denoise"
    compileSdk = libs.versions.compileSdk.get().toInt()
    defaultConfig {
        minSdk = libs.versions.minSdk.get().toInt()
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

dependencies {
    implementation(libs.kotlinx.coroutines.android)
    // LiteRT dependency — added when inference code is written:
    // implementation(libs.litert)

    testImplementation(libs.junit)
}
