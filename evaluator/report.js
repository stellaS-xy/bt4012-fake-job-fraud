(() => {
  const search = document.getElementById('search');
  const filter = document.getElementById('filter');
  const rows = [...document.querySelectorAll('.posting-row')];
  const cases = [...document.querySelectorAll('details.case')];
  function applyFilters() {
    const query = search.value.trim().toLowerCase();
    const action = filter.value;
    let visible = 0;
    rows.forEach((row, i) => {
      const show = row.dataset.search.includes(query) && (action === 'all' || row.dataset.action === action);
      row.hidden = !show;
      cases[i].hidden = !show;
      if (show) visible++;
    });
    document.getElementById('visible-count').textContent = `${visible} of ${rows.length} shown`;
    document.getElementById('no-results').hidden = visible !== 0;
  }
  search.addEventListener('input', applyFilters);
  filter.addEventListener('change', applyFilters);
  document.getElementById('expand-all').addEventListener('click', () => cases.filter(c => !c.hidden).forEach(c => c.open = true));
  document.getElementById('collapse-all').addEventListener('click', () => cases.forEach(c => {
    c.open = false;
    c.querySelectorAll('details').forEach(child => child.open = false);
  }));
  document.querySelectorAll('[data-open]').forEach(link => link.addEventListener('click', () => {
    document.getElementById(link.dataset.open).open = true;
  }));
  function openHash() {
    const current = document.getElementById(location.hash.slice(1));
    if (current?.classList.contains('case')) current.open = true;
  }
  window.addEventListener('hashchange', openHash);
  openHash();
})();
