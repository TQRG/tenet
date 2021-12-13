// fnLoc >= targetLoc
function isContained(fnLoc, targetLoc) {
  const {sline, scol, eline, ecol} = targetLoc
  const {start, end} = fnLoc


  const startOkay = start.line < sline || (start.line === sline && start.column <= scol)
  const endOkay = end.line > eline || (end.line === eline && end.column >= ecol)

  return startOkay && endOkay
}


module.exports = function(fileInfo, api, options) {
  const j = api.jscodeshift
  const {loc} = options

  const tokens = loc.split(',')
  const targetLoc = {
    sline: parseInt(tokens[0]),
    scol: parseInt(tokens[1]),
    eline: parseInt(tokens[2]),
    ecol: parseInt(tokens[3])
  }
  let currentSource = j(fileInfo.source, { tabWidth: 1 })

  let result = ""

  currentSource.find(j.FunctionExpression)
    .forEach(p => {
      const fnLoc = p.value.loc
      if (fnLoc && isContained(fnLoc, targetLoc)) {
        const {start, end} = fnLoc
        result = `${start.line},${start.column},${end.line},${end.column}`
      }
    })

  if (result !== "") return result

  currentSource.find(j.FunctionDeclaration)
    .forEach(p => {

      const fnLoc = p.value.loc
      if (fnLoc && isContained(fnLoc, targetLoc)) {
        const {start, end} = fnLoc
        result = `${start.line},${start.column},${end.line},${end.column}`
      }
    })

  return result
}
