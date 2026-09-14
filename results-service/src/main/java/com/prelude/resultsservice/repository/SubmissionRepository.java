package com.prelude.resultsservice.repository;

import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

import com.prelude.resultsservice.entity.SubmissionEntity;

public interface SubmissionRepository extends JpaRepository<SubmissionEntity, UUID> {
}