document.addEventListener('DOMContentLoaded', () => {
  const $ = id => document.getElementById(id);
  let sequence = 0;
  const capabilities = {refrigerated_transport:'Refrigerated transport', fragile_handling:'Fragile-item handling', signature_on_delivery:'Signature on delivery'};
  function addQuote(data = {}, open = false) {
    sequence += 1;
    const usedIds = new Set([...$('custom-quotes').querySelectorAll('[data-field="id"]')].map(input => input.value));
    while (usedIds.has(`Q${sequence}`)) sequence += 1;
    const card = document.createElement('details');
    card.className = 'quote-card';
    card.open = open;
    const summary = document.createElement('summary'); card.appendChild(summary);
    const fieldsContainer = document.createElement('div'); fieldsContainer.className = 'field-grid'; card.appendChild(fieldsContainer);
    const id = document.createElement('input'); id.type = 'hidden'; id.dataset.field = 'id'; id.value = data.id || `Q${sequence}`; fieldsContainer.appendChild(id);
    const fields = [
      ['carrier', 'Carrier', 'text', data.carrier || ''],
      ['method', 'Transport method', 'text', data.method || ''],
      ['quoted_total_usd', 'All-in price (USD)', 'number', data.quoted_total_usd ?? ''],
      ['arrival_date', 'Arrival date', 'date', data.arrival_date || ''],
      ['capacity_units', 'Capacity (units)', 'number', data.capacity_units ?? ''],
      ['terms', 'Complete supplier terms', 'textarea', data.terms || ''],
    ];
    for (const [name, labelText, type, value] of fields) {
      const label = document.createElement('label'); label.textContent = labelText;
      if (name === 'terms') label.className = 'wide-field';
      const input = document.createElement(type === 'textarea' ? 'textarea' : 'input');
      input.dataset.field = name; input.required = true;
      if (type === 'textarea') input.rows = 3;
      else input.type = type;
      if (type === 'number') { input.min = '0'; input.step = name === 'capacity_units' ? '1' : '0.01'; }
      input.value = value; label.appendChild(input); fieldsContainer.appendChild(label);
    }
    const insuredLabel = document.createElement('label'); insuredLabel.className = 'checkbox-label';
    const insured = document.createElement('input'); insured.type = 'checkbox'; insured.dataset.field = 'insurance_included';
    insured.checked = data.insurance_included ?? true;
    insuredLabel.append(insured, document.createTextNode('Transit insurance included')); fieldsContainer.appendChild(insuredLabel);
    for (const [key, name] of Object.entries(capabilities)) {
      const label = document.createElement('label'); label.textContent = name + ' available';
      const select = document.createElement('select'); select.dataset.capability = key;
      for (const [value, text] of [['unknown','Unknown / not stated'],['yes','Yes — stated by supplier'],['no','No — not provided']]) {
        const option = document.createElement('option'); option.value = value; option.textContent = text; select.appendChild(option);
      }
      label.appendChild(select); fieldsContainer.appendChild(label);
    }
    const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'secondary-btn'; remove.textContent = 'Remove quotation';
    remove.addEventListener('click', () => {
      if ($('custom-quotes').children.length === 1) return;
      card.remove(); $('custom-fields').dispatchEvent(new Event('input', {bubbles: true}));
    });
    card.appendChild(remove); $('custom-quotes').appendChild(card);
  }
  function refreshForm() {
    const required = [...$('custom-fields').querySelectorAll('[data-requirement]:checked')].map(input => input.dataset.requirement);
    const cards = [...$('custom-quotes').children];
    let missing = 0;
    for (const card of cards) {
      const name = card.querySelector('[data-field="carrier"]').value.trim();
      const price = card.querySelector('[data-field="quoted_total_usd"]').value;
      card.querySelector('summary').textContent = `${card.querySelector('[data-field="id"]').value} · ${name || 'New quotation'}${price === '' ? '' : ' · $' + Number(price).toLocaleString()}`;
      card.querySelector('button').disabled = cards.length === 1;
      for (const select of card.querySelectorAll('[data-capability]')) {
        const relevant = required.includes(select.dataset.capability);
        select.closest('label').hidden = !relevant;
        if (relevant && select.value === 'unknown') missing += 1;
      }
    }
    $('quote-count').textContent = `${cards.length} quote${cards.length === 1 ? '' : 's'}`;
    $('add-custom-quote').disabled = cards.length >= 12;
    const other = $('custom-other-requirements').value.trim();
    $('custom-review-summary').textContent = other
      ? 'Other essential conditions need manual review. Confirming this form will not bypass that check.'
      : missing ? `${missing} required supplier answer${missing === 1 ? ' is' : 's are'} still unknown. Fill them in under Supplier quotes, or the run will stop for review.`
      : `${cards.length} quotes · Budget, deadline, capacity${$('custom-insurance').checked ? ' and insurance' : ''}${required.length ? ' · ' + required.map(key => capabilities[key]).join(', ') : ''}. Check these against the complete request and supplier terms.`;
  }
  function read() {
    const options = [...$('custom-quotes').children].map(card => {
      const data = {};
      for (const input of card.querySelectorAll('[data-field]')) {
        data[input.dataset.field] = input.type === 'checkbox' ? input.checked : input.type === 'number' ? Number(input.value) : input.value;
      }
      return data;
    });
    const quoteCapabilities = {};
    for (const card of $('custom-quotes').children) {
      const id = card.querySelector('[data-field="id"]').value;
      quoteCapabilities[id] = Object.fromEntries([...card.querySelectorAll('[data-capability]')].map(input => [input.dataset.capability, input.value === 'unknown' ? null : input.value === 'yes']));
    }
    return {title: $('custom-title').value, brief: $('custom-brief').value, route: $('custom-route').value,
      quantity: Number($('custom-quantity').value), budget_usd: Number($('custom-budget').value),
      deadline: $('custom-deadline').value, require_insurance: $('custom-insurance').checked, options,
      requirement_review: {confirmed:$('custom-requirements-confirmed').checked,
        required_capabilities:[...$('custom-fields').querySelectorAll('[data-requirement]:checked')].map(input=>input.dataset.requirement),
        other_requirements:$('custom-other-requirements').value.split('\n').map(line=>line.trim()).filter(Boolean),
        quote_capabilities:quoteCapabilities}};
  }
  function fill(data) {
    for (const [id, key] of [['title','title'],['brief','brief'],['route','route'],['quantity','quantity'],['budget','budget_usd'],['deadline','deadline']]) $('custom-' + id).value = data[key];
    $('custom-insurance').checked = data.require_insurance;
    $('custom-quotes').replaceChildren(); sequence = 0; data.options.forEach((quote, index) => addQuote(quote, index === 0));
    const review = data.requirement_review || {};
    $('custom-fields').querySelectorAll('[data-requirement]').forEach(input=> { input.checked = (review.required_capabilities || []).includes(input.dataset.requirement); });
    $('custom-other-requirements').value = (review.other_requirements || []).join('\n');
    for (const card of $('custom-quotes').children) {
      const answers = review.quote_capabilities?.[card.querySelector('[data-field="id"]').value] || {};
      card.querySelectorAll('[data-capability]').forEach(input=> { input.value = answers[input.dataset.capability] == null ? 'unknown' : answers[input.dataset.capability] ? 'yes' : 'no'; });
    }
    $('custom-requirements-confirmed').checked = false;
    $('custom-fields').dispatchEvent(new Event('input', {bubbles: true}));
  }
  $('add-custom-quote').addEventListener('click', () => {
    if ($('custom-quotes').children.length < 12) { addQuote({}, true); $('custom-fields').dispatchEvent(new Event('input', {bubbles: true})); }
  });
  $('load-custom-template').addEventListener('click', async () => {
    const status = $('custom-template-status'); status.textContent = 'Loading example…';
    try {
      const response = await fetch('/api/compare/template/' + $('custom-template').value);
      if (!response.ok) throw new Error('Example unavailable');
      fill(await response.json()); status.textContent = 'Example loaded. Edit the fields to create your scenario.';
    } catch (error) { status.textContent = error.message; }
  });
  addQuote({}, true); addQuote();
  $('custom-fields').addEventListener('input', event => {
    if (event.target !== $('custom-requirements-confirmed')) $('custom-requirements-confirmed').checked = false;
    refreshForm();
  });
  // Native validation must reveal required fields inside collapsed sections.
  $('run-form').addEventListener('invalid', event => {
    for (let parent = event.target.parentElement; parent; parent = parent.parentElement) {
      if (parent.tagName === 'DETAILS') parent.open = true;
    }
  }, true);
  refreshForm();
  window.CustomScenario = {read, fill};
});
