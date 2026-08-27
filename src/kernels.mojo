"""Vectorized backtesting kernels exposed through a C ABI."""

from std.math import floor, isnan, sqrt
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def nan_value() -> Float64:
    return 0.0 / Float64(0)


def infinity() -> Float64:
    return 1.0 / Float64(0)


def get_return(input_value: Float64, output_value: Float64) -> Float64:
    if input_value == 0.0:
        if output_value == 0.0:
            return 0.0
        if isnan(output_value):
            return output_value
        return infinity() if output_value > 0.0 else -infinity()
    var value = (output_value - input_value) / input_value
    return -value if input_value < 0.0 else value


def shift_transform(
    values: FPtr,
    dst: FPtr,
    rows: Int,
    cols: Int,
    n: Int,
    fill_value: Float64,
    op: Int,
):
    for i in range(rows):
        for col in range(cols):
            var k = i * cols + col
            if op == 0:
                dst[k] = values[(i - n) * cols + col] if i - n >= 0 else fill_value
            elif op == 1:
                dst[k] = values[(i + n) * cols + col] if i + n < rows else fill_value
            elif i < n:
                dst[k] = nan_value()
            elif op == 2:
                dst[k] = values[k] - values[(i - n) * cols + col]
            else:
                dst[k] = values[k] / values[(i - n) * cols + col] - 1.0


def rolling_moments(
    values: FPtr,
    dst: FPtr,
    rows: Int,
    cols: Int,
    window: Int,
    minp: Int,
    ddof: Int,
    take_std: Bool,
):
    for col in range(cols):
        var total = 0.0
        var mean = 0.0
        var moment2 = 0.0
        var valid = 0
        for i in range(rows):
            var value = values[i * cols + col]
            if not isnan(value):
                valid += 1
                var delta = value - mean
                mean += delta / Float64(valid)
                moment2 += delta * (value - mean)
                total += value
            if i >= window:
                var old = values[(i - window) * cols + col]
                if not isnan(old):
                    total -= old
                    if valid == 1:
                        mean = 0.0
                        moment2 = 0.0
                    else:
                        var new_count = valid - 1
                        var new_mean = (Float64(valid) * mean - old) / Float64(new_count)
                        moment2 -= (old - mean) * (old - new_mean)
                        mean = new_mean
                    valid -= 1
            var k = i * cols + col
            if valid < minp:
                dst[k] = nan_value()
            elif not take_std:
                dst[k] = total / Float64(valid)
            elif valid <= ddof:
                dst[k] = nan_value()
            else:
                var variance = moment2
                if variance < 0.0:
                    variance = 0.0
                dst[k] = sqrt(variance / Float64(valid - ddof))


def rolling_extreme(
    values: FPtr,
    dst: FPtr,
    queue: IPtr,
    rows: Int,
    cols: Int,
    window: Int,
    minp: Int,
    take_max: Bool,
):
    for col in range(cols):
        var head = 0
        var tail = 0
        var valid = 0
        for i in range(rows):
            if i >= window:
                var old = values[(i - window) * cols + col]
                if not isnan(old):
                    valid -= 1
            var first = i - window + 1
            while head < tail and Int(queue[head]) < first:
                head += 1
            var value = values[i * cols + col]
            if not isnan(value):
                valid += 1
                while head < tail:
                    var previous = values[Int(queue[tail - 1]) * cols + col]
                    var remove = previous <= value if take_max else previous >= value
                    if not remove:
                        break
                    tail -= 1
                queue[tail] = Int64(i)
                tail += 1
            var k = i * cols + col
            if valid < minp or head == tail:
                dst[k] = nan_value()
            else:
                dst[k] = values[Int(queue[head]) * cols + col]


def returns_transform(
    values: FPtr, init_values: FPtr, dst: FPtr, rows: Int, cols: Int
):
    for col in range(cols):
        var input_value = init_values[col]
        for i in range(rows):
            var k = i * cols + col
            var output_value = values[k]
            dst[k] = get_return(input_value, output_value)
            input_value = output_value


def cumulative_transform(
    returns: FPtr,
    dst: FPtr,
    rows: Int,
    cols: Int,
    start_value: Float64,
    drawdown: Bool,
):
    for col in range(cols):
        var cumulative = 1.0
        var peak = 1.0
        for i in range(rows):
            var k = i * cols + col
            if not isnan(returns[k]):
                cumulative *= 1.0 + returns[k]
            if i == 0 or cumulative > peak:
                peak = cumulative
            if drawdown:
                dst[k] = cumulative / peak - 1.0
            elif start_value == 0.0:
                dst[k] = cumulative - 1.0
            else:
                dst[k] = cumulative * start_value


def nan_mean(values: FPtr, rows: Int, cols: Int, col: Int, offset: Float64 = 0.0) -> Float64:
    var total = 0.0
    var count = 0
    for i in range(rows):
        var value = values[i * cols + col]
        if not isnan(value):
            total += value - offset
            count += 1
    return total / Float64(count) if count > 0 else nan_value()


def nan_std(
    values: FPtr,
    rows: Int,
    cols: Int,
    col: Int,
    ddof: Int,
    offset: Float64 = 0.0,
) -> Float64:
    var mean = nan_mean(values, rows, cols, col, offset)
    if isnan(mean):
        return mean
    var moment2 = 0.0
    var count = 0
    for i in range(rows):
        var value = values[i * cols + col]
        if not isnan(value):
            var delta = value - offset - mean
            moment2 += delta * delta
            count += 1
    if count <= ddof:
        return nan_value()
    return sqrt(moment2 / Float64(count - ddof))


def sharpe_ratio_matrix(
    returns: FPtr,
    result: FPtr,
    rows: Int,
    cols: Int,
    ann_factor: Float64,
    risk_free: Float64,
    ddof: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var col = 0
    while col + W <= cols:
        var total = SIMD[DType.float64, W](0.0)
        var count = SIMD[DType.float64, W](0.0)
        for i in range(rows):
            var value = returns.load[width=W](i * cols + col)
            var valid = value.eq(value)
            total += valid.select(value - risk_free, SIMD[DType.float64, W](0.0))
            count += valid.select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        var mean = total / count
        var moment2 = SIMD[DType.float64, W](0.0)
        for i in range(rows):
            var value = returns.load[width=W](i * cols + col)
            var valid = value.eq(value)
            var delta = value - risk_free - mean
            moment2 += valid.select(
                delta * delta, SIMD[DType.float64, W](0.0)
            )
        for lane in range(W):
            if count[lane] <= Float64(ddof):
                result[col + lane] = nan_value()
            elif moment2[lane] == 0.0:
                result[col + lane] = infinity()
            else:
                var deviation = sqrt(moment2[lane] / (count[lane] - Float64(ddof)))
                result[col + lane] = mean[lane] / deviation * sqrt(ann_factor)
        col += W
    while col < cols:
        result[col] = single_metric(
            returns, rows, cols, col, 5, ann_factor, risk_free, 0.0, ddof
        )
        col += 1


def cumulative_final(
    returns: FPtr, rows: Int, cols: Int, col: Int, start_value: Float64
) -> Float64:
    var product = 1.0
    for i in range(rows):
        var value = returns[i * cols + col]
        if not isnan(value):
            product *= 1.0 + value
    return product - 1.0 if start_value == 0.0 else product * start_value


def max_drawdown(returns: FPtr, rows: Int, cols: Int, col: Int) -> Float64:
    if rows == 0:
        return nan_value()
    var cumulative = 1.0
    var peak = 1.0
    var worst = 0.0
    for i in range(rows):
        var value = returns[i * cols + col]
        if not isnan(value):
            cumulative *= 1.0 + value
        if i == 0 or cumulative > peak:
            peak = cumulative
        var dd = cumulative / peak - 1.0
        if dd < worst:
            worst = dd
    return worst


def single_metric(
    returns: FPtr,
    rows: Int,
    cols: Int,
    col: Int,
    op: Int,
    p1: Float64,
    p2: Float64,
    p3: Float64,
    i1: Int,
) -> Float64:
    if op == 0:
        if rows == 0:
            return nan_value()
        return cumulative_final(returns, rows, cols, col, 1.0) ** (p1 / Float64(rows)) - 1.0
    if op == 1:
        if rows < 2:
            return nan_value()
        return nan_std(returns, rows, cols, col, i1) * p1 ** (1.0 / p2)
    if op == 2:
        return max_drawdown(returns, rows, cols, col)
    if op == 3:
        var dd = max_drawdown(returns, rows, cols, col)
        if dd == 0.0:
            return nan_value()
        var annual = cumulative_final(returns, rows, cols, col, 1.0) ** (p1 / Float64(rows)) - 1.0
        return annual / abs(dd)
    if op == 4:
        if p1 <= -1.0:
            return nan_value()
        var threshold = p2 if p1 == 1.0 else (1.0 + p2) ** (1.0 / p1) - 1.0
        var gains = 0.0
        var losses = 0.0
        for i in range(rows):
            var adjusted = returns[i * cols + col] - p3 - threshold
            if adjusted > 0.0:
                gains += adjusted
            elif adjusted < 0.0:
                losses -= adjusted
        return infinity() if losses == 0.0 else gains / losses
    if op == 5:
        if rows < 2:
            return nan_value()
        var mean = nan_mean(returns, rows, cols, col, p2)
        var deviation = nan_std(returns, rows, cols, col, i1, p2)
        if deviation == 0.0:
            return infinity()
        return mean / deviation * sqrt(p1)
    if op == 6 or op == 7:
        if op == 7 and rows < 2:
            return nan_value()
        var sumsq = 0.0
        var total = 0.0
        var count = 0
        for i in range(rows):
            var value = returns[i * cols + col]
            if not isnan(value):
                var adjusted = value - p2
                total += adjusted
                if adjusted < 0.0:
                    sumsq += adjusted * adjusted
                count += 1
        if count == 0:
            return nan_value()
        var risk = sqrt(sumsq / Float64(count)) * sqrt(p1)
        if op == 6:
            return risk
        if risk == 0.0:
            return infinity()
        return (total / Float64(count) * p1) / risk
    return cumulative_final(returns, rows, cols, col, p1)


def pair_metric(
    returns: FPtr,
    benchmark: FPtr,
    rows: Int,
    cols: Int,
    col: Int,
    op: Int,
    ann_factor: Float64,
    risk_free: Float64,
    ddof: Int,
) -> Float64:
    if rows < 2:
        return nan_value()
    if op == 0:
        var total = 0.0
        var count = 0
        for i in range(rows):
            var a = returns[i * cols + col]
            var b = benchmark[i * cols + col]
            if not isnan(a) and not isnan(b):
                total += a - b
                count += 1
        if count == 0:
            return nan_value()
        var mean = total / Float64(count)
        var moment2 = 0.0
        for i in range(rows):
            var a = returns[i * cols + col]
            var b = benchmark[i * cols + col]
            if not isnan(a) and not isnan(b):
                var delta = a - b - mean
                moment2 += delta * delta
        if count <= ddof:
            return nan_value()
        var deviation = sqrt(moment2 / Float64(count - ddof))
        return infinity() if deviation == 0.0 else mean / deviation

    var benchmark_total = 0.0
    var pair_count = 0
    for i in range(rows):
        var a = returns[i * cols + col]
        var b = benchmark[i * cols + col]
        if not isnan(a) and not isnan(b):
            benchmark_total += b
            pair_count += 1
    if pair_count == 0:
        return nan_value()
    var benchmark_mean = benchmark_total / Float64(pair_count)
    var covariance = 0.0
    var variance = 0.0
    for i in range(rows):
        var a = returns[i * cols + col]
        var b = benchmark[i * cols + col]
        if not isnan(a) and not isnan(b):
            var residual = b - benchmark_mean
            covariance += residual * a
            variance += residual * residual
    covariance /= Float64(pair_count)
    variance /= Float64(pair_count)
    if variance < 1.0e-30:
        return nan_value()
    var beta = covariance / variance
    if op == 1:
        return beta
    var alpha_total = 0.0
    for i in range(rows):
        var a = returns[i * cols + col]
        var b = benchmark[i * cols + col]
        if not isnan(a) and not isnan(b):
            alpha_total += (a - risk_free) - beta * (b - risk_free)
    return (alpha_total / Float64(pair_count) + 1.0) ** ann_factor - 1.0


def less_for_selection(a: Float64, b: Float64) -> Bool:
    if isnan(b):
        return not isnan(a)
    if isnan(a):
        return False
    return a < b


def select_kth(values: FPtr, n: Int, kth: Int) -> Float64:
    var left = 0
    var right = n - 1
    while left < right:
        var middle = left + (right - left) // 2
        var a = values[left]
        var b = values[middle]
        var c = values[right]
        if less_for_selection(b, a):
            var tmp = a
            a = b
            b = tmp
        if less_for_selection(c, b):
            b = c
            if less_for_selection(b, a):
                b = a
        var pivot = b
        var lower = left
        var i = left
        var upper = right
        while i <= upper:
            if less_for_selection(values[i], pivot):
                var tmp = values[lower]
                values[lower] = values[i]
                values[i] = tmp
                lower += 1
                i += 1
            elif less_for_selection(pivot, values[i]):
                var tmp = values[upper]
                values[upper] = values[i]
                values[i] = tmp
                upper -= 1
            else:
                i += 1
        if kth < lower:
            right = lower - 1
        elif kth > upper:
            left = upper + 1
        else:
            return values[kth]
    return values[left]


def percentile_select(values: FPtr, n: Int, q: Float64) -> Float64:
    if n == 0:
        return nan_value()
    var position = q * Float64(n - 1)
    var lower = Int(floor(position))
    if lower >= n - 1:
        return select_kth(values, n, n - 1)
    var fraction = position - Float64(lower)
    if fraction == 0.0:
        return select_kth(values, n, lower)
    var lower_value = 0.0
    var upper_value = 0.0
    if lower < n // 2:
        upper_value = select_kth(values, n, lower + 1)
        lower_value = values[0]
        for i in range(1, lower + 1):
            if less_for_selection(lower_value, values[i]):
                lower_value = values[i]
    else:
        lower_value = select_kth(values, n, lower)
        upper_value = values[lower + 1]
        for i in range(lower + 2, n):
            if less_for_selection(values[i], upper_value):
                upper_value = values[i]
    return lower_value * (1.0 - fraction) + upper_value * fraction


def quantile_metric(
    returns: FPtr,
    scratch: FPtr,
    rows: Int,
    cols: Int,
    col: Int,
    op: Int,
    cutoff: Float64,
) -> Float64:
    var count = 0
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= rows:
        var value = (returns + i * cols + col).strided_load[width=W](cols)
        if op == 2:
            scratch.store(i, value)
            count += W
        else:
            var valid = value.eq(value)
            if valid.reduce_and():
                scratch.store(count, value)
                count += W
            else:
                for lane in range(W):
                    if valid[lane]:
                        scratch[count] = value[lane]
                        count += 1
        i += W
    while i < rows:
        var value = returns[i * cols + col]
        if op == 2 or not isnan(value):
            scratch[count] = value
            count += 1
        i += 1
    if count == 0:
        return nan_value()
    if op == 0:
        var right = abs(percentile_select(scratch, count, 0.95))
        var left = abs(percentile_select(scratch, count, 0.05))
        return infinity() if left == 0.0 else right / left
    if op == 1:
        return percentile_select(scratch, count, cutoff)
    var cutoff_index = Int(Float64(rows - 1) * cutoff)
    _ = select_kth(scratch, count, cutoff_index)
    var total = 0.0
    for i in range(cutoff_index + 1):
        total += scratch[i]
    return total / Float64(cutoff_index + 1)


def clean_signals(
    entries: BPtr,
    exits: BPtr,
    entries_dst: BPtr,
    exits_dst: BPtr,
    rows: Int,
    cols: Int,
    entry_first: Bool,
):
    for col in range(cols):
        var phase = -1
        for i in range(rows):
            var k = i * cols + col
            entries_dst[k] = 0
            exits_dst[k] = 0
            if entries[k] != 0 and exits[k] != 0:
                continue
            if entries[k] != 0 and (phase == -1 or phase == 0):
                phase = 1
                entries_dst[k] = 1
            if exits[k] != 0 and ((not entry_first and phase == -1) or phase == 1):
                phase = 0
                exits_dst[k] = 1


@export("mvt_get_return")
def mvt_get_return(input_value: Float64, output_value: Float64) abi("C") -> Float64:
    return get_return(input_value, output_value)


@export("mvt_shift")
def mvt_shift(
    values: Int,
    dst: Int,
    rows: Int,
    cols: Int,
    n: Int,
    fill_value: Float64,
    op: Int,
) abi("C"):
    shift_transform(fp(values), fp(dst), rows, cols, n, fill_value, op)


@export("mvt_rolling_moments")
def mvt_rolling_moments(
    values: Int,
    dst: Int,
    rows: Int,
    cols: Int,
    window: Int,
    minp: Int,
    ddof: Int,
    take_std: Int,
) abi("C"):
    rolling_moments(
        fp(values), fp(dst), rows, cols, window, minp, ddof, take_std != 0
    )


@export("mvt_rolling_extreme")
def mvt_rolling_extreme(
    values: Int,
    dst: Int,
    queue: Int,
    rows: Int,
    cols: Int,
    window: Int,
    minp: Int,
    take_max: Int,
) abi("C"):
    rolling_extreme(
        fp(values), fp(dst), ip(queue), rows, cols, window, minp, take_max != 0
    )


@export("mvt_returns")
def mvt_returns(
    values: Int, init_values: Int, dst: Int, rows: Int, cols: Int
) abi("C"):
    returns_transform(fp(values), fp(init_values), fp(dst), rows, cols)


@export("mvt_cumulative")
def mvt_cumulative(
    returns: Int,
    dst: Int,
    rows: Int,
    cols: Int,
    start_value: Float64,
    drawdown: Int,
) abi("C"):
    cumulative_transform(
        fp(returns), fp(dst), rows, cols, start_value, drawdown != 0
    )


@export("mvt_single_metric")
def mvt_single_metric(
    returns: Int,
    dst: Int,
    rows: Int,
    cols: Int,
    op: Int,
    p1: Float64,
    p2: Float64,
    p3: Float64,
    i1: Int,
) abi("C"):
    var result = fp(dst)
    var source = fp(returns)
    if op == 5:
        sharpe_ratio_matrix(source, result, rows, cols, p1, p2, i1)
        return
    for col in range(cols):
        result[col] = single_metric(source, rows, cols, col, op, p1, p2, p3, i1)


@export("mvt_pair_metric")
def mvt_pair_metric(
    returns: Int,
    benchmark: Int,
    dst: Int,
    rows: Int,
    cols: Int,
    op: Int,
    ann_factor: Float64,
    risk_free: Float64,
    ddof: Int,
) abi("C"):
    var result = fp(dst)
    var source = fp(returns)
    var reference = fp(benchmark)
    for col in range(cols):
        result[col] = pair_metric(
            source, reference, rows, cols, col, op, ann_factor, risk_free, ddof
        )


@export("mvt_quantile_metric")
def mvt_quantile_metric(
    returns: Int,
    scratch: Int,
    dst: Int,
    rows: Int,
    cols: Int,
    op: Int,
    cutoff: Float64,
) abi("C"):
    var result = fp(dst)
    var source = fp(returns)
    var work = fp(scratch)
    for col in range(cols):
        result[col] = quantile_metric(
            source, work + col * rows, rows, cols, col, op, cutoff
        )


@export("mvt_clean_signals")
def mvt_clean_signals(
    entries: Int,
    exits: Int,
    entries_dst: Int,
    exits_dst: Int,
    rows: Int,
    cols: Int,
    entry_first: Int,
) abi("C"):
    clean_signals(
        bp(entries),
        bp(exits),
        bp(entries_dst),
        bp(exits_dst),
        rows,
        cols,
        entry_first != 0,
    )
