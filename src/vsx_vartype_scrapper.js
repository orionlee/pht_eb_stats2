// Utility to get the list of VSX variable types from
// https://www.aavso.org/vsx/index.php?view=about.vartypes
//
// Usage:
// 1. Open a browser tab to the above VSX list page
// 2. Copy-and-paste the script in the page's developer tool console.
// 3. The list is printed out as a tab-delimited txt that one can copy and save
//
// Note: the output TSV contains the type groups, variable types and their descriptions
// One need to process it further manually to make it a proper mapping table.

function getVSXTypes() {
  const types = Array.from(document.querySelectorAll('h2 ~ table a[name]'), el => {
    if (el.parentElement.tagName == 'H3') {
      // Case main group
      return [el.textContent, "", ""];
    }
    if (el.parentElement.tagName == "B") {
      // case a specific type
      const type = el.textContent;
      const isNonGCVS = (() => {
        const gcvsEl = el.parentElement.nextElementSibling;
        return (
          gcvsEl &&
          gcvsEl.tagName == 'IMG' &&
          gcvsEl?.src == 'https://www.aavso.org/vsx/_images/non-gcvs.gif'
        ) ? 'T' : 'F';
      })();

      let desc = el.parentElement.parentElement.parentElement?.nextElementSibling?.textContent || '';
      desc = desc.replace(/[\n\r\t]/g, ' ');

      return [type, isNonGCVS, desc];
    }
    // unexpected, print them for debug
    return [el.textContent, "?", `Unhandled node. parent type: ${el.parentElement.tagName}`];
  });
  return types;
}


function vsxTypesAsTableStr(delimiter = "\t") {
  const vsxTypes = getVSXTypes();
  return vsxTypes
    .map(type => type.join(delimiter))
    .join("\n");
}


console.log(vsxTypesAsTableStr());
