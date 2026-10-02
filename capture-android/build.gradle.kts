// Root build script: declares plugin versions for subprojects; not a module itself.
buildscript {
    dependencies {
        // AGP 9.x enables built-in Kotlin (bundled KGP 2.2.10). Adding a higher KGP
        // version here is the documented way to upgrade the built-in Kotlin version.
        // Must stay in sync with the `kotlin` version in gradle/libs.versions.toml.
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.4.0")
    }
}

plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
}
