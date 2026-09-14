package com.prelude.resultsservice.service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.prelude.resultsservice.api.PreludeApiException;
import com.prelude.resultsservice.dto.TestSetDtos.AddImagesRequest;
import com.prelude.resultsservice.dto.TestSetDtos.Candidate;
import com.prelude.resultsservice.dto.TestSetDtos.CheckRequest;
import com.prelude.resultsservice.dto.TestSetDtos.CheckResponse;
import com.prelude.resultsservice.dto.TestSetDtos.ManifestImage;
import com.prelude.resultsservice.dto.TestSetDtos.ManifestResponse;
import com.prelude.resultsservice.entity.HeldoutTestImageEntity;

import jakarta.persistence.EntityManager;
import jakarta.persistence.PersistenceContext;

@Service
public class TestSetService {

    @PersistenceContext
    private EntityManager em;

    @Transactional(readOnly = true)
    public ManifestResponse manifest() {
        List<HeldoutTestImageEntity> all = em
                .createQuery("select h from HeldoutTestImageEntity h order by h.imageId", HeldoutTestImageEntity.class)
                .getResultList();
        List<ManifestResponse.Image> images = all.stream()
                .map(h -> new ManifestResponse.Image(h.getImageId(), h.getSha256(), h.getSource(), h.getManifestVersion()))
                .toList();
        return new ManifestResponse(images.size(), images);
    }

    @Transactional(readOnly = true)
    public CheckResponse check(CheckRequest request) {
        List<HeldoutTestImageEntity> all = em
                .createQuery("select h from HeldoutTestImageEntity h", HeldoutTestImageEntity.class)
                .getResultList();
        Map<String, HeldoutTestImageEntity> byId = new HashMap<>();
        Map<String, HeldoutTestImageEntity> byHash = new HashMap<>();
        for (HeldoutTestImageEntity h : all) {
            byId.put(h.getImageId(), h);
            byHash.put(h.getSha256().toLowerCase(Locale.ROOT), h);
        }

        List<CheckResponse.Match> heldout = new ArrayList<>();
        List<Candidate> notInHeldout = new ArrayList<>();
        for (Candidate c : request.candidates()) {
            boolean hasId = c.imageId() != null && !c.imageId().isBlank();
            boolean hasHash = c.sha256() != null && !c.sha256().isBlank();
            if (!hasId && !hasHash) {
                throw new PreludeApiException.ContractViolation("INCOMPLETE_CANDIDATE",
                        "Each candidate must carry at least one of imageId or sha256.");
            }
            HeldoutTestImageEntity match = null;
            String matchedBy = null;
            if (hasId && byId.containsKey(c.imageId())) {
                match = byId.get(c.imageId());
                matchedBy = "imageId";
            } else if (hasHash && byHash.containsKey(c.sha256().toLowerCase(Locale.ROOT))) {
                match = byHash.get(c.sha256().toLowerCase(Locale.ROOT));
                matchedBy = "sha256";
            }
            if (match != null) {
                heldout.add(new CheckResponse.Match(c, match.getImageId(), matchedBy));
            } else {
                notInHeldout.add(c);
            }
        }
        return new CheckResponse(heldout, notInHeldout);
    }

    @Transactional
    public ManifestResponse addImages(AddImagesRequest request) {
        Integer maxVersion = em
                .createQuery("select max(h.manifestVersion) from HeldoutTestImageEntity h", Integer.class)
                .getSingleResult();
        int version = (maxVersion == null ? 0 : maxVersion) + 1;

        for (ManifestImage img : request.images()) {
            String hash = img.sha256().trim().toLowerCase(Locale.ROOT);
            if (em.find(HeldoutTestImageEntity.class, img.imageId()) != null) {
                throw new PreludeApiException.ContractViolation("MANIFEST_CONFLICT",
                        "imageId already locked in the held-out manifest: " + img.imageId());
            }
            Long hashCount = em
                    .createQuery("select count(h) from HeldoutTestImageEntity h where h.sha256 = :hash", Long.class)
                    .setParameter("hash", hash)
                    .getSingleResult();
            if (hashCount > 0) {
                throw new PreludeApiException.ContractViolation("MANIFEST_CONFLICT",
                        "sha256 already locked in the held-out manifest (submitted for imageId " + img.imageId() + ")");
            }
            em.persist(new HeldoutTestImageEntity(img.imageId(), hash, img.source(), version, Instant.now()));
        }
        em.flush();
        return manifest();
    }
}