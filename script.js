document.addEventListener('DOMContentLoaded', () => {
  const $ = id => document.getElementById(id);
  const runs = new Map();
  let scenarios = [];
  let reports = [];
  let currentReport = 0;
  let busy = false;
  const number = value => new Intl.NumberFormat().format(value);
  const duration = seconds => seconds < 60 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const compressed = () => document.querySelector('input[name="mode"]:checked').value === 'compressed';
  const fidelity = () => $('fidelity-select').value;
  const requestKey = () => JSON.stringify([$('scenario-select').value, $('scenario-select').value === 'custom' ? window.CustomScenario.read() : $('scenario-select').value === 'legacy_custom' ? $('custom-prompt').value.trim() : '', isPaired() ? fidelity() : 'legacy']);
  const selectedMode = () => compressed() ? 'compressed' : 'baseline';
  const isPaired = () => $('scenario-select').value === 'custom' || scenarios.find(s => s.id === $('scenario-select').value)?.kind === 'paired';
  const runLabel = () => isPaired() ? 'Run comparison' : 'Run experiment';
  function updateModelLabel() {
    const model = runs.get(requestKey())?.experiment?.configuration.model || scenarios.find(s => s.id === $('scenario-select').value)?.model || 'llama3';
    $('runtime-model').textContent = `Ollama / ${model}`;
  }

  function acceptPair(data, key) {
    if (data.status === "execution_failed") throw new Error(`Experiment failed: ${data.error}. The failed attempt and measured costs were saved.`);
    if (data.status === 'needs_review') {
      if (!Array.isArray(data.requirement_validation?.issues)) throw new Error('Missing requirement review details.');
      runs.set(key, {experiment:data}); return;
    }
    for (const mode of ['baseline', 'compressed']) {
      if (!data[mode]?.reports || !Number.isFinite(data[mode].token_usage) || !data[mode].evaluation) throw new Error('Incomplete paired result.');
    }
    if (data.baseline.shared_context_sha256 !== data.compressed.shared_context_sha256) throw new Error('The results do not share the same upstream context.');
    if (['validated-v11-adaptive', 'validated-v12-tools', 'validated-v13-requirements'].includes(data.configuration.pipeline_version) && (!data.context_audit || !Number.isFinite(data.compression_only?.token_usage))) throw new Error('Missing adaptive audit or compression-only measurement.');
    runs.set(key, {baseline: data.baseline, compressed: data.compressed, experiment: data});
  }

  function tableRow(target, cells) {
    const row = document.createElement('tr');
    cells.forEach(value => { const cell = document.createElement('td'); cell.textContent = value; row.appendChild(cell); });
    target.appendChild(row);
  }

  function setStatus(text, kind = '') {
    $('status').textContent = text;
    $('status').className = `status ${kind}`;
  }
  function renderComparison() {
    updateModelLabel();
    const pair = runs.get(requestKey()) || {};
    $('requirement-review-result').hidden = pair.experiment?.status !== 'needs_review';
    $('comparison-body').replaceChildren();
    if (pair.experiment?.status === 'needs_review') {
      const review = pair.experiment.requirement_validation;
      $('requirement-review-issues').replaceChildren();
      review.issues.forEach(issue => { const item = document.createElement('li'); item.textContent = issue; $('requirement-review-issues').appendChild(item); });
      $('requirement-review-scope').textContent = review.scope;
      $('empty-state').hidden = $('report-view').hidden = true;
      $('token-val').textContent = '0'; $('time-val').textContent = '—';
      $('token-note').textContent = 'Stopped before inference';
      $('delta-val').textContent = 'Not measured'; $('delta-val').className = 'metric-value';
      $('delta-subtitle').textContent = 'No decision or token saving claimed';
      $('result-mode').textContent = 'NEEDS REVIEW';
      $('workflow-note').textContent = 'Requirements need review. No LLM agents ran.';
      $('paired-details').hidden = $('attribution-note').hidden = true;
      $('download-pair').hidden = false;
      $('comparison-description').textContent = 'Requirements must be resolved before the measured comparison can run.';
      $('comparison-note').textContent = pair.experiment.measurement_note;
      tableRow($('comparison-body'), ['Requirement review', '0', '—', 'Not evaluated', 'Needs review']);
      setStatus('Needs review'); return;
    }
    const comparisonRows = [['baseline', 'Full context']];
    if (pair.experiment?.compression_only) comparisonRows.push(['compression_only', 'Compression only']);
    comparisonRows.push(['compressed', ['validated-v12-tools','validated-v13-requirements'].includes(pair.experiment?.configuration.pipeline_version) ? 'Compression + Python tools' : fidelity() === 'adaptive' && isPaired() ? 'Adaptive compressor' : 'Compressed']);
    for (const [mode, label] of comparisonRows) {
      const result = pair[mode] || pair.experiment?.[mode];
      const row = document.createElement('tr');
      [mode === 'compressed' && result?.compression?.applied === false ? (result.compression.mode === 'auto' ? 'Automatic · skipped' : 'Compression rejected') : label, result ? number(result.token_usage) : '—', result ? duration(result.execution_time_seconds) : '—', result?.evaluation ? `${result.evaluation.passed}/${result.evaluation.total}` : '—', result ? (result.status === 'failed_validation' ? 'Failed validation' : result.status === 'validated' ? 'Validated' : 'Historical result') : 'Not run yet'].forEach((text, index) => {
        const cell = document.createElement('td');
        cell.textContent = text;
        if (index === 4 && result) cell.className = result.status === 'failed_validation' ? 'negative' : 'complete';
        if (index === 3 && result?.evaluation) cell.className = result.evaluation.all_passed ? 'positive' : 'negative';
        row.appendChild(cell);
      });
      $('comparison-body').appendChild(row);
    }
    $('delta-val').textContent = '—';
    $('delta-val').className = 'metric-value';
    $('delta-subtitle').textContent = isPaired() ? 'Run a comparison to measure savings' : 'Run both modes for this scenario';
    if (pair.baseline && pair.compressed && pair.baseline.token_usage > 0) {
      const saved = pair.baseline.token_usage - pair.compressed.token_usage;
      $('delta-val').textContent = `${(100 * saved / pair.baseline.token_usage).toFixed(1)}%`;
      $('delta-val').classList.add(saved >= 0 ? 'positive' : 'negative');
      $('delta-subtitle').textContent = `${number(Math.abs(saved))} tokens ${saved >= 0 ? 'fewer' : 'more'} than full context`;
    }
    if (pair.compressed?.compression?.applied === false && pair.baseline) {
      $('delta-val').textContent = 'Skipped';
      $('delta-val').className = 'metric-value';
      $('delta-subtitle').textContent = `No compression savings claimed. Run-to-run difference: ${number(pair.baseline.token_usage - pair.compressed.token_usage)} tokens.`;
    }
    // Show difference_attribution when we have a paired result
    const attr = pair.experiment?.comparison?.difference_attribution;
    if (attr) {
      const labels = {
        compression_routing_and_output_variation: 'Adaptive saving combines compression, agent routing and output variation. The compression-only branch measures the first contribution.',
        paired_compression_and_output_variation: 'Compression applied — savings include context reduction and output variation.',
        rejected_compression_cost_and_output_variation: 'Compressor ran but fell back to full context — its cost is charged; difference is output variation only.',
        run_to_run_variation_no_compression: 'No compression — token difference is ordinary run-to-run output variation.',
      };
      $('attribution-label').textContent = labels[attr] || attr;
      $('attribution-note').hidden = false;
    } else {
      $('attribution-note').hidden = true;
    }
    const experiment = pair.experiment;
    const skippedRoles = compressed() ? experiment?.comparison?.skipped_llm_roles || [] : [];
    $('workflow-note').textContent = skippedRoles.length ? `In this result, ${skippedRoles.join(', ')} are Python checks. ${6 - skippedRoles.length} LLM agents ran, including Ingestion.` : fidelity() === 'adaptive' && isPaired() ? 'Six workflow roles. Adaptive routing can use Python for Finance, Compliance and, on long inputs, Logistics.' : '';
    $('paired-details').hidden = $('download-pair').hidden = !experiment;
    $('comparison-description').textContent = isPaired()
      ? fidelity() === 'adaptive'
        ? 'Three measured branches: full source, compression only, and compression with adaptive routing.'
        : fidelity() === 'complete'
        ? 'Complete-source comparison: every brief and supplier term remains in the compressed branch.'
        : 'Decision-focused comparison: selected caveats and protected facts reach downstream agents.'
      : 'Latest successful run in each mode, for the selected request.';
    $('comparison-note').textContent = isPaired()
      ? fidelity() === 'adaptive'
        ? 'Same supplied facts and Llama settings. Adaptive routing replaces structured specialist checks with Python tools. Risk and Decision retain every supplier term. Agent counts differ; this is a workflow comparison.'
        : 'Paired experiment: same source facts, agent roles, and model settings. The Ingestion call is shared; later agents run in each branch. All compressor tokens count.'
      : 'Separate runs may generate different reports. Token reduction alone does not measure output quality.';
    if (experiment) {
      const summary = experiment.comparison;
      if (experiment.status === 'failed_validation' || !(summary.all_branches_validated ?? summary.both_checklists_passed)) $('delta-subtitle').textContent += ' · quality checks failed';
      const compressor = experiment.compressed.calls.find(call => call.role === 'Compressor');
      const decisionsPassed = summary.all_branches_validated ?? summary.both_checklists_passed;
      $('paired-summary').textContent = `${Number.isFinite(summary.review_input_reduction_percent) ? summary.review_input_reduction_percent.toFixed(1) + "% less input for the Risk reviewer." : "Risk input reduction unavailable."} Compressor cost: ${number(compressor?.total_tokens ?? 0)} tokens. ${decisionsPassed ? 'All measured decisions passed the supplied-fact checks.' : 'At least one decision failed checks; inspect the checklist before drawing conclusions.'} Model: ${experiment.configuration.model}. Recorded ${new Date(experiment.created_at).toLocaleString()}.${experiment.compressed.compression?.summary_truncated ? ' Summary reached its output limit; incomplete trailing text was omitted and source facts retained.' : ''}`;
      if (Number.isFinite(summary.downstream_input_reduction_percent)) {
        $('paired-summary').textContent += ` Across all downstream agents: ${summary.downstream_input_reduction_percent.toFixed(1)}% input reduction. Net savings = ${number(summary.downstream_input_tokens_saved)} input tokens + ${number(summary.downstream_output_tokens_saved)} output tokens − ${number(summary.compressor_tokens)} compressor tokens. Output length changes are measured separately from context reduction.`;
      }
      if (experiment.compressed.compression?.applied === false) $('paired-summary').textContent += ` Compression skipped: ${experiment.compressed.compression.reason}.`;
      const formatWarnings = [...experiment.shared.calls, ...experiment[selectedMode()].calls]
        .filter(call => call.report_format && (!call.report_format.sections_match || !call.report_format.within_word_target))
        .map(call => call.role);
      if (formatWarnings.length) $('paired-summary').textContent += ` Report format or length targets missed: ${formatWarnings.join(', ')}. Original output is preserved.`;
      if (summary.truncated_calls?.length) $('paired-summary').textContent += ` Output limits reached: ${summary.truncated_calls.join(', ')}. Inspect these reports for incomplete content.`;
      $('paired-summary').textContent += ' Checks cover quote selection and supplied facts; they do not establish full report quality.';
      if (['validated-v11-adaptive', 'validated-v12-tools', 'validated-v13-requirements'].includes(experiment.configuration.pipeline_version)) {
        $('paired-summary').textContent += ` Adaptive path: ${summary.adaptive_path}. Compression-only contribution: ${number(summary.compression_only_saved_tokens)} tokens (${summary.compression_only_reduction_percent.toFixed(1)}%). Additional routing contribution: ${number(summary.routing_saved_tokens)} tokens (${number(summary.routing_input_tokens_saved)} input + ${number(summary.routing_output_tokens_saved)} output). Source audit: ${experiment.context_audit?.all_checks_passed ? 'passed' : 'FAILED'}. All three decisions: ${summary.all_branches_validated ? 'validated' : 'FAILED'}.`;
        $('paired-summary').textContent += summary.skipped_llm_roles.length ? ` Python replaced these LLM calls: ${summary.skipped_llm_roles.join(', ')}.` : ' Specialist views can omit prose; Risk and Decision receive the complete brief and terms.';
        $('paired-summary').textContent += ` Experiment cost including all three branches: ${number(summary.total_experiment_tokens)} tokens. No LLM compressor call was used. Checks do not prove arbitrary conditions in prose.`;
      }
      else if (experiment.configuration.pipeline_version === 'validated-v10-lossless-verified') $('paired-summary').textContent += ' Complete-source mode: every original brief, carrier, method, and supplier term is copied into every agent prompt with one compact Python audit. No LLM compressor call is billed.';
      else if (experiment.configuration.pipeline_version === 'validated-v5-graph') $('paired-summary').textContent += ' Decision-focused mode: some supplier prose is omitted. Use the complete-source mode when full wording matters.';
      else $('paired-summary').textContent += ' Historical protocol: rerun to use the revised measurements and evaluator.';
      const guard = experiment[selectedMode()].claim_metrics;
      if (experiment.requirement_validation) $('paired-summary').textContent += ' Requirement scope: ' + experiment.requirement_validation.scope;
      if (guard) $('paired-summary').textContent += ` Generated claim tuples correct: ${guard.claims_correct}/${guard.claims_total}. Rejected items: ${guard.rejected_items}; malformed reports: ${guard.schema_failures}. Rejected prose was not forwarded.`;
      const policy = experiment.compressed.compression;
      if (Number.isFinite(policy?.estimated_net_tokens_saved)) $('paired-summary').textContent += ` Predicted net saving: ${number(policy.estimated_net_tokens_saved)} tokens. Calibration pairs: ${policy.calibration?.sample_count ?? 'historical'}.`;
      if (experiment.status === 'failed_validation') $('paired-summary').textContent += ' FAILED VALIDATION: at least one final selection or explanation was blocked.';
      $('quality-body').replaceChildren();
      const labels = {structured_answer:'Structured answer', correct_verdict:'Correct proceed / hold ruling', lowest_cost_feasible_quote:'Lowest-cost feasible quote', quoted_total_correct:'Quoted total preserved', arrival_correct:'Arrival date preserved', quantity_preserved:'Shipment quantity preserved', budget_preserved:'Budget preserved', deadline_preserved:'Deadline preserved', insurance_requirement_met:'Insurance requirement satisfied', delivery_requirement_met:'Selected quote meets deadline', budget_requirement_met:'Selected quote fits budget', capacity_requirement_met:'Selected quote carries all units'};
      Object.keys(experiment.baseline.evaluation.checks).forEach(check => tableRow($('quality-body'), [labels[check] || check, ...['baseline','compressed'].map(mode => experiment[mode].evaluation.checks[check] ? 'Pass' : 'Fail')]));
      $('token-breakdown').replaceChildren();
      const allCalls = mode => [...experiment.shared.calls, ...experiment[mode].calls];
      for (const role of ['Ingestion','Logistics','Finance','Compressor','Risk','Compliance','Decision']) {
        const cells = [role];
        for (const mode of ['baseline','compressed']) {
          const call = allCalls(mode).find(item => item.role === role);
          const pythonRole = mode === 'compressed' && summary.skipped_llm_roles?.includes(role);
          cells.push(call ? number(call.input_tokens) : pythonRole ? 'Python · 0 LLM tokens' : '—', call ? number(call.output_tokens) : '—');
        }
        tableRow($('token-breakdown'), cells);
      }
    }
  }

  // Build text and limited formatting as DOM nodes; model output never becomes HTML.
  function inlineText(target, text) {
    text.split(/(\*\*[^*]+\*\*)/g).forEach(part => {
      if (part.startsWith('**') && part.endsWith('**')) {
        const strong = document.createElement('strong');
        strong.textContent = part.slice(2, -2);
        target.appendChild(strong);
      } else target.appendChild(document.createTextNode(part));
    });
  }
  function renderReport() {
    const target = $('report-content');
    target.replaceChildren();
    target.scrollTop = 0;
    target.setAttribute('aria-labelledby', `report-tab-${currentReport}`);
    const clean = reports[currentReport][1].replace(/\[(?:MAP_)?IMAGE:\s*[^\]]*\]/g, '').trim();
    let list = null;
    clean.split(/\r?\n/).forEach(line => {
      const text = line.trim();
      if (!text) { list = null; return; }
      const bullet = text.match(/^([-*]|\d+\.)\s+(.+)/);
      if (bullet) {
        const tag = /\d/.test(bullet[1]) ? 'OL' : 'UL';
        if (!list || list.tagName !== tag) { list = document.createElement(tag); target.appendChild(list); }
        const item = document.createElement('li');
        inlineText(item, bullet[2]); list.appendChild(item);
      } else {
        list = null;
        const heading = text.match(/^#{1,6}\s+(.+)/);
        const paragraph = document.createElement(heading ? 'h3' : 'p');
        inlineText(paragraph, heading ? heading[1] : text);
        target.appendChild(paragraph);
      }
    });
    if (!clean) target.textContent = 'No report returned.';
    document.querySelectorAll('[role="tab"]').forEach((tab, index) => {
      tab.setAttribute('aria-selected', String(index === currentReport));
      tab.tabIndex = index === currentReport ? 0 : -1;
    });
    $('copy-report').textContent = 'Copy report';
  }
  function showResult(result, mode) {
    const compressionShown = mode === 'compressed' && result.compression?.applied !== false;
    $('compression-step').hidden = $('compression-arrow').hidden = !compressionShown;
    $('empty-state').hidden = true;
    $('report-view').hidden = false;
    $('token-val').textContent = number(result.token_usage);
    $('time-val').textContent = duration(result.execution_time_seconds);
    $('token-note').textContent = result.compression?.applied === false ? 'Compression skipped · ' + result.compression.reason : mode === 'compressed' ? 'Includes the compression step' : 'Across all workflow agents';
    $('result-mode').textContent = result.compression?.applied === false ? 'FULL CONTEXT / COMPRESSION SKIPPED' : mode === 'compressed' ? 'COMPRESSED' : 'FULL CONTEXT';
    reports = [
      ['Final decision', result.reports.Decision],
      ['Ingestion', result.reports.Ingestion],
      ['Logistics', result.reports.Logistics],
      ['Finance', result.reports.Finance],
      ['Risk', result.reports.Risk],
      ['Compliance', result.reports.Compliance],
    ];
    const replacedRoles = mode === 'compressed' ? runs.get(requestKey())?.experiment?.comparison?.skipped_llm_roles || [] : [];
    reports = reports.map(([label, text]) => [replacedRoles.includes(label) ? label + ' · Python checks' : label, text]);
    if (mode === 'compressed' && result.reports.Compressor) reports.push(['Compressed context', result.reports.Compressor]);
    if (result.compression) reports.push(['Compression policy', JSON.stringify(result.compression, null, 2)]);
    if (result.status !== 'failed_validation' && result.evaluation?.decision && Object.keys(result.evaluation.decision).length) {
      const decision = result.evaluation.decision;
      reports[0][1] = [
        `## Final decision: ${decision.verdict || 'Unspecified'}`,
        `**Decision checklist:** ${result.evaluation.passed}/${result.evaluation.total}${result.evaluation.all_passed ? ' supplied-fact checks passed.' : ' — failed checks; this result needs review.'}`,
        `**Quote:** ${decision.option_id ?? 'None'}`,
        `**Quoted total:** ${decision.total_cost_usd == null ? 'Not applicable' : '$' + number(decision.total_cost_usd)}`,
        `**Arrival:** ${decision.arrival_date ?? 'Not applicable'}`,
        `**Quantity:** ${decision.quantity ?? 'Not stated'} units`,
        `**Budget:** $${number(decision.budget_usd)}`,
        `**Deadline:** ${decision.deadline ?? 'Not stated'}`,
        `## Reasoning\n${decision.reasoning || 'Not provided'}`,
        ...(Array.isArray(decision.conditions) && decision.conditions.length ? ['## Conditions', ...decision.conditions.map(condition => `- ${condition}`)] : []),
      ].join('\n\n');
    }
    if (result.status === 'failed_validation') reports.push(['Rejected decision · audit only', result.calls.find(c => c.role === 'Decision')?.output || 'Unavailable']);
    if (result.claim_metrics) reports.push(['Claim validation', JSON.stringify(result.report_validation, null, 2)]);
    if (result.structured_reports) reports.push(['Structured claims', JSON.stringify(result.structured_reports, null, 2)]);
    if (result.constraint_audit) reports.push(['Python calculations', result.constraint_audit.text]);
    if (result.report_validation) reports.push(['Python validation', JSON.stringify({
      reports: result.report_validation,
      context_routing: result.context_routing,
      protected_facts: result.compression?.validation || 'Full source context',
      decision: result.evaluation?.structure_validation,
      failed_checks: result.evaluation?.failed_checks,
    }, null, 2)]);
    const sharedContext = runs.get(requestKey())?.experiment?.shared.context;
    if (sharedContext) reports.push(['Shared source context', sharedContext]);
    const protectedRecord = runs.get(requestKey())?.experiment?.shared.protected_facts;
    if (protectedRecord) reports.push(['Protected facts', JSON.stringify(protectedRecord, null, 2)]);
    const adaptiveExperiment = runs.get(requestKey())?.experiment;
    if (adaptiveExperiment?.context_audit) {
      reports.push(['Context integrity audit', JSON.stringify(adaptiveExperiment.context_audit, null, 2)]);
      reports.push(['Routing plan and source coverage', JSON.stringify(adaptiveExperiment.configuration.adaptive_plan, null, 2)]);
    }
    if (adaptiveExperiment?.requirement_validation) reports.push(['Requirement review', JSON.stringify(adaptiveExperiment.requirement_validation, null, 2)]);
    $('report-tabs').replaceChildren();
    reports.forEach(([label], index) => {
      const button = document.createElement('button');
      button.type = 'button'; button.role = 'tab'; button.id = `report-tab-${index}`;
      button.setAttribute('aria-controls', 'report-content'); button.textContent = label;
      button.addEventListener('click', () => { currentReport = index; renderReport(); });
      button.addEventListener('keydown', event => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        currentReport = event.key === 'Home' ? 0 : event.key === 'End' ? reports.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + reports.length) % reports.length;
        renderReport(); $(`report-tab-${currentReport}`).focus();
      });
      $('report-tabs').appendChild(button);
    });
    currentReport = 0; renderReport();
  }
  function updateSelection() {
    if (busy) return;
    const isCustom = $('scenario-select').value === 'custom';
    const isLegacyCustom = $('scenario-select').value === 'legacy_custom';
    $('custom-prompt-group').hidden = !isLegacyCustom;
    $('custom-fields').hidden = $('custom-fields').disabled = !isCustom;
    document.querySelector('.workspace-grid').classList.toggle('custom-layout', isCustom);
    $('scenario-details').hidden = isCustom || isLegacyCustom;
    $('setup-summary').textContent = isPaired()
      ? fidelity() === 'adaptive' ? 'Llama 3 · Automatic optimization' : fidelity() === 'complete' ? 'Llama 3 · Compression only' : 'Llama 3 · Experimental optimization'
      : 'Legacy experiment · Unverified free text';
    $('fidelity-group').hidden = !isPaired();
    $('fidelity-note').textContent = fidelity() === 'adaptive'
      ? 'Compressor + Python tools: two specialist roles use Python on short inputs, three on long inputs. Every active LLM reviewer keeps the complete source. Savings are measured, not guaranteed.'
      : fidelity() === 'complete'
      ? 'Complete source preserves every supplied field and term. It usually saves fewer tokens.'
      : 'Decision-focused context can save more tokens, but omits some source wording. Final quote facts are validated.';
    $('custom-prompt').required = isLegacyCustom;
    $('scenario-preview').textContent = isCustom ? '' : scenarios.find(s => s.id === $('scenario-select').value)?.prompt || '';
    $('compression-step').hidden = $('compression-arrow').hidden = !compressed();
    $('error').hidden = true;
    $('run-label').textContent = runLabel();
    $('load-saved').hidden = !isPaired() || isCustom;
    $('mode-legend').textContent = isPaired() ? 'Report view (both workflows run)' : 'Context mode';
    $('run-hint').textContent = isPaired()
      ? 'One click compares the original and optimized workflows, checks the decisions, and measures token use.'
      : 'Legacy free-text workflow: no protected-fact checks or paired accounting. Separate runs can differ for reasons other than compression.';
    $('run-caption').textContent = isPaired() ? 'Runs the measured comparison. Allow several minutes.' : 'Local model runs can take a few minutes.';
    const result = runs.get(requestKey())?.[selectedMode()];
    if (result) { showResult(result, selectedMode()); setStatus(result.status === 'failed_validation' ? 'Failed validation' : 'Run complete', result.status === 'failed_validation' ? 'failed' : 'complete'); }
    else {
      $('empty-state').hidden = false; $('report-view').hidden = true;
      $('token-val').textContent = $('time-val').textContent = '—';
      $('token-note').textContent = 'Across all workflow agents';
      $('result-mode').textContent = '02 / OUTPUT'; setStatus('Ready to run');
    }
    renderComparison();
  }
  $('scenario-select').addEventListener('change', updateSelection);
  $('fidelity-select').addEventListener('change', () => { updateSelection(); loadEvaluation(); });
  $('custom-prompt').addEventListener('input', updateSelection);
  $('custom-fields').addEventListener('input', updateSelection);
  document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener('change', updateSelection));
  $('copy-report').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(reports[currentReport][1]); $('copy-report').textContent = 'Copied'; }
    catch { $('copy-report').textContent = 'Select text to copy'; }
  });
  $('load-saved').addEventListener('click', async () => {
    if (busy) return;
    busy = true; $('run-fields').disabled = true;
    setStatus('Loading saved comparison');
    try {
      const response = await fetch(`/api/compare/latest/${encodeURIComponent($('scenario-select').value)}?fidelity=${encodeURIComponent(fidelity())}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not load the saved comparison.');
      acceptPair(data, requestKey());
      document.querySelector('input[name="mode"][value="compressed"]').checked = true;
      busy = false; updateSelection();
      if (data.status !== 'needs_review') setStatus(data.status === 'failed_validation' ? 'Failed validation' : 'Saved comparison', data.status === 'failed_validation' ? 'failed' : 'complete');
    } catch (error) {
      $('error').textContent = error.message; $('error').hidden = false; setStatus('No comparison loaded', 'failed');
    } finally { busy = false; $('run-fields').disabled = false; }
  });
  $('download-pair').addEventListener('click', () => {
    const experiment = runs.get(requestKey())?.experiment;
    if (!experiment) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(experiment, null, 2)], {type:'application/json'}));
    const link = document.createElement('a'); link.href = url; link.download = `${experiment.scenario.id}-${experiment.experiment_id}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  $('run-form').addEventListener('submit', async event => {
    event.preventDefault();
    if (busy) return;
    const key = requestKey(), mode = selectedMode(), pairedRun = isPaired();
    const body = { scenario_id: $('scenario-select').value, use_compression: mode === 'compressed' };
    if (body.scenario_id === 'legacy_custom') {
      body.scenario_id = 'custom';
      body.custom_prompt = $('custom-prompt').value.trim();
      if (!body.custom_prompt) { $('custom-prompt').focus(); return; }
    }
    const pairedBody = {scenario_id: $('scenario-select').value, fidelity: fidelity()};
    if (pairedBody.scenario_id === 'custom') pairedBody.custom_scenario = window.CustomScenario.read();
    busy = true; $('run-fields').disabled = true; $('run-label').textContent = pairedRun ? 'Comparing workflows…' : 'Running experiment…';
    $('error').hidden = true; $('empty-state').hidden = true; $('report-view').hidden = true; $('running-state').hidden = false;
    $('requirement-review-result').hidden = true;
    $('token-val').textContent = $('time-val').textContent = '—';
    $('delta-val').textContent = '—'; $('delta-val').className = 'metric-value';
    $('delta-subtitle').textContent = 'Measuring token use…';
    setStatus('Running', 'running');
    const started = Date.now();
    const tick = () => { const seconds = Math.floor((Date.now() - started) / 1000); $('elapsed').textContent = `Elapsed ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`; };
    tick(); const timer = setInterval(tick, 1000);
    try {
      const response = await fetch(pairedRun ? '/api/compare' : '/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(pairedRun ? pairedBody : body) });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || data.detail || `Request failed (${response.status}).`);
      if (pairedRun) {
        acceptPair(data, key);
        if (data.status === 'needs_review') { renderComparison(); return; }
        document.querySelector('input[name="mode"][value="compressed"]').checked = true;
        $('compression-step').hidden = $('compression-arrow').hidden = false;
        showResult(data.compressed, 'compressed'); renderComparison(); setStatus(data.status === 'failed_validation' ? 'Failed validation' : 'Comparison complete', data.status === 'failed_validation' ? 'failed' : 'complete');
        return;
      }
      const requiredReports = ['Ingestion', 'Logistics', 'Finance', 'Risk', 'Compliance', 'Decision'];
      if (mode === 'compressed') requiredReports.push('Compressor');
      if (!Number.isFinite(data.token_usage) || !Number.isFinite(data.execution_time_seconds) || !data.reports || !requiredReports.every(name => typeof data.reports[name] === 'string')) throw new Error('The server returned an incomplete result. Please try again.');
      const pair = runs.get(key) || {}; pair[mode] = data; runs.set(key, pair);
      showResult(data, mode); renderComparison(); setStatus(data.status === 'failed_validation' ? 'Failed validation' : 'Run complete', data.status === 'failed_validation' ? 'failed' : 'complete');
    } catch (error) {
      $('error').textContent = `Run failed. ${error.message || 'Check the server and Ollama connection, then try again.'}`;
      $('error').hidden = false; $('empty-state').hidden = false;
      setStatus('Run failed', 'failed');
    } finally {
      clearInterval(timer); busy = false; $('running-state').hidden = true; $('run-fields').disabled = false;
      $('run-label').textContent = runLabel();
    }
  });
  async function loadScenarios() {
    try {
      const response = await fetch('/api/scenarios');
      if (!response.ok) throw new Error('Unable to load scenarios.');
      scenarios = await response.json();
      if (!Array.isArray(scenarios)) throw new Error('Invalid scenario list.');
      // Historical workflows remain available in the backend, outside the dashboard.
      scenarios = scenarios.filter(scenario => scenario.kind === 'paired');
      $('scenario-select').replaceChildren();
      const examples = document.createElement('optgroup'); examples.label = 'Ready-to-run examples';
      scenarios.forEach((scenario, index) => {
        const option = document.createElement('option'); option.value = scenario.id;
        option.textContent = scenario.title || `${index + 1}. ${scenario.domain || scenario.id}`;
        examples.appendChild(option);
      });
      const custom = document.createElement('option'); custom.value = 'custom'; custom.textContent = 'Create your own scenario…';
      $('scenario-select').append(examples, custom); $('run-fields').disabled = false; updateSelection();
    } catch {
      setStatus('Connection unavailable', 'failed');
      $('error').textContent = 'Could not load scenarios. Check that the server is running, then reload this page.'; $('error').hidden = false;
    }
  }
  async function loadEvaluation() {
    try {
      const response = await fetch(`/api/evaluation/latest?fidelity=${encodeURIComponent(fidelity())}`);
      if (!response.ok) throw new Error('No saved model results for this protocol yet.');
      const data = await response.json();
      $('evaluation-description').textContent = `${data.phase === 'pilot' ? 'Pilot results; broader repeated tests are still needed.' : data.phase === 'repeat_check' ? 'Small repeated check.' : data.phase === 'evaluate' ? 'Repeated evaluation' : data.phase === 'benchmark' ? 'Model benchmark' : 'Stress test'} · protocol ${data.protocol} · recorded ${new Date(data.created_at).toLocaleString()}. ${data.scope || ''}`;
      $('evaluation-body').replaceChildren();
      for (const [model, stats] of Object.entries(data.models)) {
        if (!stats.attempted_pairs) continue;
        tableRow($('evaluation-body'), [model, stats.attempted_pairs, stats.needs_review_pairs ?? 0,
          `${stats.validated_decisions}/${stats.attempted_decisions ?? 2 * stats.attempted_pairs}`,
          stats.factual_claim_accuracy == null ? 'No valid claims' : `${(stats.factual_claim_accuracy * 100).toFixed(1)}%`,
          `${stats.rejected_items} / ${stats.schema_failures}`,
          `${stats.compression_applied_pairs} / ${stats.skipped_pairs}`,
          stats.negative_savings_pairs != null ? stats.negative_savings_pairs : '—',
          stats.mean_applied_reduction_percent == null ? 'Not measured' : `${stats.mean_applied_reduction_percent.toFixed(1)}% ± ${stats.sd_applied_reduction_percent?.toFixed(1) ?? '—'}`,
          stats.mean_wall_seconds == null ? '—' : duration(stats.mean_wall_seconds)]);
      }
    } catch (error) { $('evaluation-body').replaceChildren(); $('evaluation-description').textContent = error.message; }
  }
  loadEvaluation();
  renderComparison(); loadScenarios();
});
