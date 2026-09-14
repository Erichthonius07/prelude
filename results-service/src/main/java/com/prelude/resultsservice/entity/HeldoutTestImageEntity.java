package com.prelude.resultsservice.entity;

import java.time.Instant;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "heldout_test_image")
public class HeldoutTestImageEntity {

    @Id
    @Column(name = "image_id")
    private String imageId;

    @Column(name = "sha256", nullable = false, length = 64)
    private String sha256;

    @Column(name = "source", nullable = false)
    private String source;

    @Column(name = "manifest_version", nullable = false)
    private int manifestVersion;

    @Column(name = "added_at", nullable = false)
    private Instant addedAt;

    protected HeldoutTestImageEntity() {
    }

    public HeldoutTestImageEntity(String imageId, String sha256, String source, int manifestVersion, Instant addedAt) {
        this.imageId = imageId;
        this.sha256 = sha256;
        this.source = source;
        this.manifestVersion = manifestVersion;
        this.addedAt = addedAt;
    }

    public String getImageId() {
        return imageId;
    }

    public String getSha256() {
        return sha256;
    }

    public String getSource() {
        return source;
    }

    public int getManifestVersion() {
        return manifestVersion;
    }
}