package com.prelude.resultsservice.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.prelude.resultsservice.entity.HeldoutTestImageEntity;

public interface HeldoutTestImageRepository extends JpaRepository<HeldoutTestImageEntity, String> {
}