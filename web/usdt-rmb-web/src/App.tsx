import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

// =====================================================================
// ALPHA · ETH 量化分析工作台（轻量分析工具版）
// ---------------------------------------------------------------------
// 设计目标：让用户在 3 秒内知道"该不该交易"。
//
// 视觉基调：
//   - 明亮 / 大量留白 / 卡片式（参考 Linear · Notion · Vercel · Stripe）
//   - 强视觉层级：Hero 决策 → 多周期 → 关键指标 → 详情归因
//   - 强交互：hover 浮起 / tooltip / loading skeleton / 可折叠 / 平滑入场动画
//
// 后端接口（不变，仍然消费 app/api_service/routes.py 的三个综合分析接口）：
//   GET  /analysis/latest          —— 最新一次综合分析
//   GET  /analysis/history         —— 最近 N 条
//   GET  /analysis/{signal_id}     —— 单条详情（含思维链 + 完整因子）
//   POST /signal/refresh           —— 主动触发一次新分析
// =====================================================================

// ---------------------------------------------------------------------
// 类型定义（与 serialize_signal_full 输出严格对齐）
// ---------------------------------------------------------------------

/** 偏向颜色：success=绿（多）/ error=红（空）/ default=黄灰（中性） */
type BiasColor = 'success' | 'error' | 'default'
/** 生命周期颜色：processing/warning/success/error/default */
type LifecycleColor = 'processing' | 'warning' | 'success' | 'error' | 'default'

/** 单个周期的方向投票（5m / 15m / 1h / 4h / 1d 等） */
interface TimeframeAlignmentItem {
  timeframe: string
  bias: 'long' | 'short' | 'neutral' | string
  label: string
  color: BiasColor
}

/** 入场区间 */
interface EntryZone {
  low: number
  high: number
  mid: number | null
  width_pct: number | null
}

/** 止盈点位 */
interface TakeProfitItem {
  level: string
  price: number
  reward_pct: number | null
}

/** 规则引擎单条因子贡献 */
interface RuleContribution {
  key: string
  timeframe: string
  group: string
  factor: string
  contribution: number
  abs_contribution: number
}

/** 完整的综合分析视图（API 响应主结构） */
interface AnalysisFull {
  signal_id: number | null
  symbol: string
  source: string
  source_label: string
  ts: string | null
  time_ago_seconds: number | null
  time_ago_human: string

  summary: {
    bias: string
    bias_label: string
    bias_color: BiasColor
    confidence: number | null
    confidence_label: string
    regime: string | null
    current_price: number | null
    risk_reward_ratio: number | null
    position_size_pct: number | null
    lifecycle_status: string | null
    lifecycle_status_label: string
    pnl_pct: number | null
    headline: string
  }

  decision: {
    bias: string
    bias_label: string
    bias_color: BiasColor
    confidence: number | null
    confidence_pct: number | null
    confidence_label: string
    reason: string
    risk: string
    suggestion: string
  }

  trading_plan: {
    has_plan: boolean
    entry_zone: EntryZone | null
    stop_loss: number | null
    stop_loss_pct: number | null
    take_profit: TakeProfitItem[]
    risk_reward_ratio: number | null
    position_size_pct: number | null
    position_size_label: string | null
  }

  timeframe_alignment: TimeframeAlignmentItem[]
  invalidation_conditions: Array<
    string | { description?: string; condition?: string }
  >

  market_context: {
    regime: string | null
    current_price: number | null
    mtf_alignment: {
      alignment_score?: number | null
      dominant_bias?: string | null
      trend_votes?: Record<string, number>
    }
    liquidations: Record<string, unknown>
    liquidity: {
      current_price?: number | null
      nearest_above_pct?: number | null
      nearest_below_pct?: number | null
      pool_above_count?: number
      pool_below_count?: number
    }
  }

  rule_engine: {
    rule_score: number | null
    top_contributions: RuleContribution[]
  }

  lifecycle: {
    status: string | null
    status_label: string
    status_color: LifecycleColor
    is_settled: boolean
    is_open: boolean
    triggered_at: string | null
    triggered_price: number | null
    exit_at: string | null
    exit_price: number | null
    pnl_pct: number | null
    max_favorable_pct: number | null
    max_adverse_pct: number | null
    expires_at: string | null
    updated_at: string | null
  }

  reasoning_available: boolean
  reasoning_total_chars: number
  reasoning_content?: string

  factors_snapshot?: Record<string, unknown>
}

/** 历史列表响应 */
interface HistoryResponse {
  symbol: string
  count: number
  items: AnalysisFull[]
}

// ---------------------------------------------------------------------
// 工具函数：格式化数字 / 百分比 / 价格 / 时间
// ---------------------------------------------------------------------

/**
 * 把 0.034 这样的小数转成 "+3.40%"。
 * - null / NaN 显示 "—"
 * - 正数加 "+"，负数原样显示（"-" 由数字本身带出）
 */
function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  const sign = v > 0 ? '+' : v < 0 ? '' : ''
  return `${sign}${(v * 100).toFixed(digits)}%`
}

/** 价格千分位 + 智能精度（>=1000 → 2 位 / >=10 → 3 位 / 否则 5 位） */
function fmtPrice(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  const digits = v >= 1000 ? 2 : v >= 10 ? 3 : 5
  return v.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

/** 简单数字：固定小数位 */
function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return v.toFixed(digits)
}

/** ISO8601 → 本地时间 "11/04 22:35:08" */
function fmtLocalTime(iso: string | null): string {
  if (!iso) return '—'
  try {
    const d = new Date(iso)
    const mm = String(d.getMonth() + 1).padStart(2, '0')
    const dd = String(d.getDate()).padStart(2, '0')
    const HH = String(d.getHours()).padStart(2, '0')
    const MI = String(d.getMinutes()).padStart(2, '0')
    const SS = String(d.getSeconds()).padStart(2, '0')
    return `${mm}/${dd} ${HH}:${MI}:${SS}`
  } catch {
    return iso
  }
}

/**
 * BiasColor → 一组语义化的 Tailwind class
 * 包括：纯文字色、淡背景（卡片高亮）、边框、纯背景（按钮 / 实心 chip）
 */
function biasPalette(color: BiasColor | string): {
  text: string
  textOn: string
  bg: string
  bgSoft: string
  border: string
  bar: string
  ring: string
} {
  switch (color) {
    case 'success':
      return {
        text: 'text-long',
        textOn: 'text-white',
        bg: 'bg-long',
        bgSoft: 'bg-long-bg',
        border: 'border-long-soft',
        bar: 'bg-long',
        ring: 'ring-long-soft',
      }
    case 'error':
      return {
        text: 'text-short',
        textOn: 'text-white',
        bg: 'bg-short',
        bgSoft: 'bg-short-bg',
        border: 'border-short-soft',
        bar: 'bg-short',
        ring: 'ring-short-soft',
      }
    default:
      return {
        text: 'text-neutral',
        textOn: 'text-white',
        bg: 'bg-neutral',
        bgSoft: 'bg-neutral-bg',
        border: 'border-neutral-soft',
        bar: 'bg-neutral',
        ring: 'ring-neutral-soft',
      }
  }
}

/** lifecycle 颜色 → 卡片 chip 用的浅色样式 */
function lifecyclePalette(c: LifecycleColor | string): string {
  switch (c) {
    case 'success':
      return 'text-long bg-long-bg border-long-soft'
    case 'error':
      return 'text-short bg-short-bg border-short-soft'
    case 'warning':
      return 'text-neutral bg-neutral-bg border-neutral-soft'
    case 'processing':
      return 'text-accent bg-accent-bg border-accent-soft'
    default:
      return 'text-muted bg-bg-2 border-hairline'
  }
}

// ---------------------------------------------------------------------
// API 客户端
// ---------------------------------------------------------------------

/** API 基础路径（vite.config.ts 已配代理到后端） */
const API_BASE = '/api'

async function apiGet<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: 'application/json' },
  })
  if (!r.ok) {
    const text = await r.text().catch(() => '')
    throw new Error(`${r.status} ${r.statusText} ${text}`.trim())
  }
  return (await r.json()) as T
}

async function apiPost<T>(path: string): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
  })
  if (!r.ok) {
    const text = await r.text().catch(() => '')
    throw new Error(`${r.status} ${r.statusText} ${text}`.trim())
  }
  return (await r.json()) as T
}

// =====================================================================
// 顶层容器 App
// =====================================================================

/**
 * 顶层应用：负责所有数据获取 + 全局状态。
 * 子组件几乎都是无状态展示组件，便于复用。
 */
function App() {
  /** 当前展示的合约（暂时写死 ETH，预留切换） */
  const [symbol] = useState<string>('ETH-USDT-SWAP')

  /** 主视图当前展示的分析（默认 latest，点击历史时切换为详情） */
  const [current, setCurrent] = useState<AnalysisFull | null>(null)
  /** 历史时间轴条目 */
  const [history, setHistory] = useState<AnalysisFull[]>([])
  /** 首屏加载中（current 还没拿到时显示骨架屏） */
  const [loading, setLoading] = useState<boolean>(true)
  /** 是否正在执行手动刷新（POST /signal/refresh 进行中） */
  const [refreshing, setRefreshing] = useState<boolean>(false)
  /** 全局错误条带文本 */
  const [error, setError] = useState<string | null>(null)
  /** 思维链抽屉是否展开 */
  const [reasoningOpen, setReasoningOpen] = useState<boolean>(false)
  /** 历史时间轴选中项 id（用于高亮） */
  const [activeId, setActiveId] = useState<number | null>(null)

  /** 拉取最新一次综合分析（带因子快照） */
  const loadLatest = useCallback(async () => {
    try {
      const data = await apiGet<AnalysisFull>(
        `/analysis/latest?symbol=${encodeURIComponent(
          symbol,
        )}&include_factors=true`,
      )
      setCurrent(data)
      setActiveId(data.signal_id ?? null)
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [symbol])

  /** 拉取历史时间轴（最近 20 条） */
  const loadHistory = useCallback(async () => {
    try {
      const data = await apiGet<HistoryResponse>(
        `/analysis/history?symbol=${encodeURIComponent(symbol)}&limit=20`,
      )
      setHistory(data.items)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [symbol])

  /** 首屏：并行拉 latest + history */
  useEffect(() => {
    let alive = true
    setLoading(true)
    Promise.all([loadLatest(), loadHistory()]).finally(() => {
      if (alive) setLoading(false)
    })
    return () => {
      alive = false
    }
  }, [loadLatest, loadHistory])


  /** 手动触发一次新分析（POST /signal/refresh），完成后刷新视图 */
  const handleRefresh = useCallback(async () => {
    setRefreshing(true)
    try {
      await apiPost(`/signal/refresh?symbol=${encodeURIComponent(symbol)}`)
      await Promise.all([loadLatest(), loadHistory()])
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setRefreshing(false)
    }
  }, [symbol, loadLatest, loadHistory])

  /** 点击历史时间轴某条 → 拉取详情（带思维链 + 完整因子） */
  const handleSelectHistory = useCallback(async (id: number) => {
    setActiveId(id)
    try {
      const data = await apiGet<AnalysisFull>(
        `/analysis/${id}?include_factors=true&include_reasoning=true&top_contributions=20`,
      )
      setCurrent(data)
    } catch (e) {
      setError((e as Error).message)
    }
  }, [])

  /** 打开思维链抽屉时，如果 current 没带 reasoning_content 就补拉一次详情 */
  const handleOpenReasoning = useCallback(async () => {
    setReasoningOpen(true)
    if (
      current?.signal_id &&
      !current.reasoning_content &&
      current.reasoning_available
    ) {
      try {
        const data = await apiGet<AnalysisFull>(
          `/analysis/${current.signal_id}?include_factors=true&include_reasoning=true&top_contributions=20`,
        )
        setCurrent(data)
      } catch (e) {
        setError((e as Error).message)
      }
    }
  }, [current])

  return (
    <div className="min-h-screen bg-bg text-ink">
      <Header
        symbol={symbol}
        current={current}
        refreshing={refreshing}
        onRefresh={handleRefresh}
      />

      {error && <ErrorBanner message={error} onClose={() => setError(null)} />}

      <main className="mx-auto max-w-[1320px] px-4 pb-24 pt-6 sm:px-6 lg:px-10">
        {loading && !current ? (
          <SkeletonHero />
        ) : current ? (
          <div className="space-y-6">
            {/* 1️⃣ 顶部决策核心（大判词 + 关键指标 + 置信度） */}
            <DecisionHero data={current} />

            {/* 2️⃣ 多周期信号（横向 tabs） */}
            <TimeframeStrip data={current.timeframe_alignment} />

            {/* 3️⃣ 二级网格：交易计划 + 生命周期 + 因子归因 + 流动性/MTF */}
            <div className="grid grid-cols-12 gap-6">
              <div className="col-span-12 xl:col-span-7">
                <TradingPlanCard data={current} />
              </div>
              <div className="col-span-12 xl:col-span-5">
                <LifecyclePanel
                  data={current.lifecycle}
                  bias={current.decision.bias}
                />
              </div>

              <div className="col-span-12 xl:col-span-7">
                <AttributionList data={current.rule_engine} />
              </div>
              <div className="col-span-12 xl:col-span-5">
                <MarketContextCard
                  liquidity={current.market_context.liquidity}
                  mtf={current.market_context.mtf_alignment}
                  regime={current.market_context.regime}
                />
              </div>

              {/* 4️⃣ Narrative 三段式 */}
              <div className="col-span-12">
                <NarrativeCards decision={current.decision} />
              </div>

              {/* 5️⃣ 历史信号时间轴 */}
              <div className="col-span-12">
                <HistoryTimeline
                  items={history}
                  activeId={activeId}
                  onSelect={handleSelectHistory}
                />
              </div>
            </div>

            {/* 6️⃣ 思维链浮动按钮 */}
            <ReasoningTrigger
              available={current.reasoning_available}
              chars={current.reasoning_total_chars}
              onOpen={handleOpenReasoning}
            />
          </div>
        ) : (
          <EmptyState onRefresh={handleRefresh} />
        )}
      </main>

      <Footer />

      {/* 思维链抽屉 */}
      <ReasoningDrawer
        open={reasoningOpen}
        onClose={() => setReasoningOpen(false)}
        content={current?.reasoning_content ?? null}
        available={current?.reasoning_available ?? false}
        totalChars={current?.reasoning_total_chars ?? 0}
      />
    </div>
  )
}

export default App

// =====================================================================
//  通用 UI 原子（高复用）
// =====================================================================

/**
 * Card：所有内容卡片的容器
 *   - 白底 + 圆角 + 极淡阴影
 *   - hover 浮起（shadow-sm → shadow-md，translate-y-[-1px]）
 *   - 可选 eyebrow / title / action / footer
 */
function Card({
  eyebrow,
  title,
  description,
  action,
  children,
  className = '',
  delay = 0,
  hoverable = true,
}: {
  /** 顶部小标题（uppercase，灰色） */
  eyebrow?: string
  /** 卡片主标题 */
  title?: string
  /** 卡片描述（标题下方一行解释） */
  description?: string
  /** 标题右侧动作（按钮 / chip 等） */
  action?: ReactNode
  children: ReactNode
  /** 额外 class */
  className?: string
  /** 入场动画延时（秒） */
  delay?: number
  /** 是否启用 hover 浮起效果 */
  hoverable?: boolean
}) {
  return (
    <section
      className={`rise-in rounded-2xl border border-hairline bg-surface p-6 shadow-card transition-all duration-200 ${
        hoverable
          ? 'hover:-translate-y-[1px] hover:shadow-card-hover hover:border-hairline-strong'
          : ''
      } ${className}`}
      style={{ animationDelay: `${delay}s` }}
    >
      {(eyebrow || title || action) && (
        <header className="mb-5 flex items-start justify-between gap-3">
          <div className="min-w-0">
            {eyebrow && (
              <div className="font-mono text-[11px] uppercase tracking-[0.16em] text-muted">
                {eyebrow}
              </div>
            )}
            {title && (
              <h3 className="mt-1 text-lg font-semibold tracking-tight text-ink">
                {title}
              </h3>
            )}
            {description && (
              <p className="mt-1 text-sm text-muted">{description}</p>
            )}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

/**
 * Badge：圆角 pill 标签
 *   - tone 控制色系：long / short / neutral / accent / muted
 *   - variant 控制 solid（实心）/ soft（淡色）/ outline（描边）
 */
type Tone = 'long' | 'short' | 'neutral' | 'accent' | 'muted'
type BadgeVariant = 'soft' | 'solid' | 'outline'

function Badge({
  children,
  tone = 'muted',
  variant = 'soft',
  size = 'sm',
  icon,
  className = '',
}: {
  children: ReactNode
  tone?: Tone
  variant?: BadgeVariant
  size?: 'xs' | 'sm' | 'md'
  icon?: ReactNode
  className?: string
}) {
  /** 用 lookup 表生成颜色，避免 Tailwind JIT 漏扫描 */
  const palettes: Record<
    Tone,
    Record<BadgeVariant, string>
  > = {
    long: {
      soft: 'bg-long-bg text-long border-long-soft',
      solid: 'bg-long text-white border-long',
      outline: 'bg-transparent text-long border-long-soft',
    },
    short: {
      soft: 'bg-short-bg text-short border-short-soft',
      solid: 'bg-short text-white border-short',
      outline: 'bg-transparent text-short border-short-soft',
    },
    neutral: {
      soft: 'bg-neutral-bg text-neutral border-neutral-soft',
      solid: 'bg-neutral text-white border-neutral',
      outline: 'bg-transparent text-neutral border-neutral-soft',
    },
    accent: {
      soft: 'bg-accent-bg text-accent border-accent-soft',
      solid: 'bg-accent text-white border-accent',
      outline: 'bg-transparent text-accent border-accent-soft',
    },
    muted: {
      soft: 'bg-bg-2 text-ink-2 border-hairline',
      solid: 'bg-ink text-white border-ink',
      outline: 'bg-transparent text-ink-2 border-hairline-strong',
    },
  }
  const sizes = {
    xs: 'h-5 px-2 text-[10px] gap-1',
    sm: 'h-6 px-2.5 text-[11px] gap-1.5',
    md: 'h-7 px-3 text-xs gap-1.5',
  }

  return (
    <span
      className={`inline-flex items-center rounded-full border font-medium ${palettes[tone][variant]} ${sizes[size]} ${className}`}
    >
      {icon}
      {children}
    </span>
  )
}

/**
 * Stat：标签 / 大数字 / 提示。最常用的指标卡片单元。
 */
function Stat({
  label,
  value,
  hint,
  tone = 'text-ink',
  size = 'md',
}: {
  label: string
  value: string
  hint?: ReactNode
  tone?: string
  size?: 'sm' | 'md' | 'lg'
}) {
  const sizes = {
    sm: 'text-lg',
    md: 'text-2xl',
    lg: 'text-3xl',
  }
  return (
    <div>
      <div className="text-[11px] font-medium uppercase tracking-wider text-muted">
        {label}
      </div>
      <div
        className={`mt-1 font-semibold tabular leading-none tracking-tight ${sizes[size]} ${tone}`}
      >
        {value}
      </div>
      {hint && (
        <div className="mt-1.5 text-[11px] leading-tight text-muted-2">
          {hint}
        </div>
      )}
    </div>
  )
}

/**
 * ProgressBar：水平进度条（也用作置信度仪表）
 * - tone 控制颜色
 * - showLabel 在右侧显示百分比
 */
function ProgressBar({
  value,
  tone = 'accent',
  className = '',
  height = 'h-2',
}: {
  /** 0-100 */
  value: number
  tone?: Tone
  className?: string
  height?: string
}) {
  const colorMap: Record<Tone, string> = {
    long: 'bg-long',
    short: 'bg-short',
    neutral: 'bg-neutral',
    accent: 'bg-accent',
    muted: 'bg-muted',
  }
  const v = Math.min(100, Math.max(0, value))
  return (
    <div
      className={`relative w-full overflow-hidden rounded-full bg-bg-2 ${height} ${className}`}
    >
      <div
        className={`absolute inset-y-0 left-0 rounded-full transition-all duration-700 ease-out ${colorMap[tone]}`}
        style={{ width: `${v}%` }}
      />
    </div>
  )
}

/**
 * Tooltip：基于 .tooltip / .tooltip-content（在 index.css 中定义）
 */
function Tooltip({
  content,
  children,
  className = '',
}: {
  content: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <span className={`tooltip ${className}`} tabIndex={0}>
      {children}
      <span className="tooltip-content">{content}</span>
    </span>
  )
}

/**
 * Skeleton：占位骨架。用于首屏 loading 状态。
 */
function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`skeleton ${className}`} />
}

/**
 * Collapsible：受控折叠容器
 * - 内置一个 "展开 / 收起" 按钮
 * - 折叠时使用 max-height 阻挡，但保留访问性
 */
function Collapsible({
  children,
  collapsedHeight = 96,
  expandLabel = '展开查看',
  collapseLabel = '收起',
}: {
  children: ReactNode
  collapsedHeight?: number
  expandLabel?: string
  collapseLabel?: string
}) {
  const [open, setOpen] = useState(false)
  return (
    <div>
      <div
        className="relative overflow-hidden transition-[max-height] duration-300 ease-out"
        style={{ maxHeight: open ? '2000px' : `${collapsedHeight}px` }}
      >
        {children}
        {!open && (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-12 bg-gradient-to-t from-surface to-transparent" />
        )}
      </div>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="mt-3 inline-flex items-center gap-1 text-[12px] font-medium text-accent hover:text-accent-2"
      >
        {open ? collapseLabel : expandLabel}
        <svg
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
          className={`transition-transform ${open ? 'rotate-180' : ''}`}
          aria-hidden
        >
          <path
            d="M3 4.5 6 7.5 9 4.5"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </div>
  )
}

// =====================================================================
//  Header / Footer / Banner / Empty / Skeleton
// =====================================================================

/**
 * 顶栏：极简 sticky 顶栏
 * - 左：品牌 + 合约
 * - 中：实时价 + live 指示灯 + 时间
 * - 右：刷新按钮
 */
function Header({
  symbol,
  current,
  refreshing,
  onRefresh,
}: {
  symbol: string
  current: AnalysisFull | null
  refreshing: boolean
  onRefresh: () => void
}) {
  const price = current?.summary.current_price ?? null
  const ago = current?.time_ago_human ?? '—'
  const live = (current?.time_ago_seconds ?? Infinity) < 120

  return (
    <header className="sticky top-0 z-30 border-b border-hairline bg-surface/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-[1320px] items-center justify-between gap-4 px-4 sm:px-6 lg:px-10">
        {/* 左：品牌 + 合约 */}
        <div className="flex min-w-0 items-center gap-4">
          <BrandMark />
          <div className="hidden h-5 w-px bg-hairline-strong sm:block" />
          <div className="hidden items-center gap-2 sm:flex">
            <span className="text-xs font-medium text-muted">合约</span>
            <span className="rounded-md bg-bg-2 px-2 py-0.5 font-mono text-xs text-ink-2">
              {symbol}
            </span>
          </div>
        </div>

        {/* 中：实时价 */}
        <div className="hidden items-center gap-3 md:flex">
          <Tooltip
            content={
              live
                ? `数据 ${ago} · 30 秒自动刷新`
                : `数据 ${ago} · 已离线，请手动刷新`
            }
          >
            <span className="flex items-center gap-1.5">
              <span
                className={`inline-block h-2 w-2 rounded-full ${
                  live ? 'pulse-dot bg-long' : 'bg-muted-2'
                }`}
              />
              <span className="text-[11px] font-medium text-muted">
                {live ? 'LIVE' : 'IDLE'}
              </span>
              <span className="text-[11px] text-muted-2">· {ago}</span>
            </span>
          </Tooltip>
          <div className="h-5 w-px bg-hairline-strong" />
          <div className="flex items-baseline gap-1">
            <span className="text-xs text-muted">$</span>
            <span className="font-display text-xl font-semibold tabular text-ink">
              {fmtPrice(price)}
            </span>
            <span className="ml-1 text-[10px] font-medium text-muted">
              USDT
            </span>
          </div>
        </div>
      </div>
    </header>
  )
}

/** 品牌图标：菱形 logo + 文字 */
function BrandMark() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-accent to-accent-2 shadow-sm">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden>
          <path d="M9 1 L17 9 L9 17 L1 9 Z" fill="white" opacity="0.95" />
          <path d="M9 5 L13 9 L9 13 L5 9 Z" fill="#2563eb" />
        </svg>
      </div>
      <div className="leading-tight">
        <div className="text-sm font-semibold tracking-tight text-ink">
          Alpha<span className="text-accent">.</span>ETH
        </div>
        <div className="text-[10px] font-medium uppercase tracking-wider text-muted">
          Analytics
        </div>
      </div>
    </div>
  )
}

/** 错误条：红色 banner，含 dismiss */
function ErrorBanner({
  message,
  onClose,
}: { 
  message: string
  onClose: () => void
}) {
  return (
    <div className="border-b border-short-soft bg-short-bg">
      <div className="mx-auto flex max-w-[1320px] items-start justify-between gap-4 px-4 py-3 sm:px-6 lg:px-10">
        <div className="flex items-start gap-2.5">
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            className="mt-0.5 shrink-0 text-short"
            aria-hidden
          >
            <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.4" />
            <path
              d="M8 4.5v3.5M8 11v.01"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          </svg>
          <div className="text-sm text-ink-2">
            <span className="mr-2 font-semibold text-short">出错</span>
            <span className="font-mono text-xs">{message}</span>
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-[11px] font-medium text-muted hover:text-ink"
        >
          关闭
        </button>
      </div>
    </div>
  )
}

/** 首屏骨架屏：三块 + 一块 hero */
function SkeletonHero() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-3 w-32" />
      <div className="rounded-2xl border border-hairline bg-surface p-8 shadow-card">
        <div className="grid grid-cols-12 gap-8">
          <div className="col-span-12 lg:col-span-7">
            <Skeleton className="h-3 w-24" />
            <Skeleton className="mt-4 h-16 w-48" />
            <Skeleton className="mt-6 h-4 w-full" />
            <Skeleton className="mt-2 h-4 w-3/4" />
          </div>
          <div className="col-span-12 lg:col-span-5 grid grid-cols-2 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i}>
                <Skeleton className="h-3 w-16" />
                <Skeleton className="mt-2 h-7 w-20" />
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="grid grid-cols-12 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="col-span-12 xl:col-span-6 rounded-2xl border border-hairline bg-surface p-6 shadow-card"
          >
            <Skeleton className="h-3 w-20" />
            <Skeleton className="mt-3 h-6 w-40" />
            <Skeleton className="mt-6 h-32 w-full" />
          </div>
        ))}
      </div>
    </div>
  )
}

/** 空态：还没生成第一条信号时显示 */
function EmptyState({ onRefresh }: { onRefresh: () => void }) {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center text-center">
      <div className="grid h-16 w-16 place-items-center rounded-full bg-accent-bg text-accent">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path
            d="M12 8v4l3 2"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          />
          <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
        </svg>
      </div>
      <h2 className="mt-6 text-2xl font-semibold tracking-tight text-ink">
        暂无分析数据
      </h2>
      <p className="mt-2 max-w-md text-sm leading-relaxed text-muted">
        分析引擎可能还在收集第一轮多周期因子。点击下方按钮主动触发，或耐心等待
        30 秒后端轮询。
      </p>
      <button
        onClick={onRefresh}
        className="mt-6 inline-flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-sm font-medium text-white shadow-card transition hover:bg-accent-2"
      >
        立即生成
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
          <path
            d="M3 7h8m0 0L7.5 3.5M11 7l-3.5 3.5"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </div>
  )
}

// =====================================================================
//  1️⃣ DecisionHero —— 顶部决策核心
// =====================================================================

/**
 * 顶部决策卡片（用户视线第一落点）
 *   - 左：大号判词（"做多" / "做空" / "观望"）+ headline + chips
 *   - 右：4 个关键指标横排（confidence / RR / position / rule score）
 *   - 下：置信度全宽进度条 + 时间戳 + bias chip
 */
function DecisionHero({ data }: { data: AnalysisFull }) {
  const { decision, summary } = data
  const palette = biasPalette(decision.bias_color)
  const conf = decision.confidence_pct ?? 0
  const tone: Tone =
    decision.bias_color === 'success'
      ? 'long'
      : decision.bias_color === 'error'
        ? 'short'
        : 'neutral'

  /** 主判词图标：上箭头 / 下箭头 / 等号 */
  const directionIcon = (
    <svg width="36" height="36" viewBox="0 0 36 36" fill="none" aria-hidden>
      {decision.bias_color === 'success' && (
        <>
          <path
            d="M18 27V9M9 18l9-9 9 9"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </>
      )}
      {decision.bias_color === 'error' && (
        <>
          <path
            d="M18 9v18M9 18l9 9 9-9"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </>
      )}
      {decision.bias_color === 'default' && (
        <>
          <path
            d="M9 14h18M9 22h18"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          />
        </>
      )}
    </svg>
  )

  return (
    <section
      className="rise-in rounded-2xl border border-hairline bg-surface p-6 shadow-card sm:p-8"
      style={{ animationDelay: '0.04s' }}
    >
      {/* 顶部 meta 行 */}
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={tone} variant="soft" size="sm">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-current" />
          {decision.bias_label}
        </Badge>
        <Badge tone="muted" variant="outline" size="sm">
          {data.source_label}
        </Badge>
        {summary.regime && (
          <Badge tone="accent" variant="soft" size="sm">
            市场状态 · {summary.regime}
          </Badge>
        )}
        <span className="ml-auto text-[11px] text-muted">
          生成时间 · {fmtLocalTime(data.ts)}
        </span>
      </div>

      <div className="mt-6 grid grid-cols-12 gap-8">
        {/* 左：判词 */}
        <div className="col-span-12 lg:col-span-7">
          <div className="flex items-center gap-4">
            <div className={`shrink-0 ${palette.text}`}>{directionIcon}</div>
            <h1
              className={`font-display text-[64px] font-bold leading-none tracking-tighter sm:text-[88px] ${palette.text}`}
            >
              {decision.bias_label}
            </h1>
          </div>
          <p className="mt-5 max-w-xl text-base leading-relaxed text-ink-2">
            {summary.headline}
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-2 text-sm">
            <span className="text-muted">当前价</span>
            <span className="font-display text-2xl font-semibold tabular text-ink">
              ${fmtPrice(summary.current_price)}
            </span>
          </div>
        </div>

        {/* 右：四象限指标 */}
        <div className="col-span-12 lg:col-span-5">
          <div className="grid grid-cols-2 gap-x-6 gap-y-5">
            <Stat
              label="置信度"
              value={`${fmtNum(conf, 1)}%`}
              tone={palette.text}
              hint={
                <Tooltip content="规则引擎 + AI 判断的综合可信度（0~100%）">
                  <span className="cursor-help underline decoration-dotted">
                    {decision.confidence_label}
                  </span>
                </Tooltip>
              }
            />
            <Stat
              label="风险收益比"
              value={fmtNum(summary.risk_reward_ratio ?? null, 2)}
              hint={
                <Tooltip content="预期止盈 / 预期止损。≥ 2 通常视为可接受">
                  <span className="cursor-help underline decoration-dotted">
                    Risk : Reward
                  </span>
                </Tooltip>
              }
            />
            <Stat
              label="建议仓位"
              value={
                summary.position_size_pct
                  ? `${(summary.position_size_pct * 100).toFixed(1)}%`
                  : '—'
              }
              hint="占可用资金"
            />
            <Stat
              label="规则评分"
              value={fmtNum(data.rule_engine.rule_score ?? null, 3)}
              tone={
                (data.rule_engine.rule_score ?? 0) > 0
                  ? 'text-long'
                  : (data.rule_engine.rule_score ?? 0) < 0
                    ? 'text-short'
                    : 'text-ink'
              }
              hint={
                (data.rule_engine.rule_score ?? 0) > 0
                  ? '看多倾向'
                  : (data.rule_engine.rule_score ?? 0) < 0
                    ? '看空倾向'
                    : '中性'
              }
            />
          </div>

          {/* 置信度全宽进度条 */}
          <div className="mt-6">
            <div className="flex items-center justify-between text-[11px] font-medium text-muted">
              <span>置信度</span>
              <span className={`tabular ${palette.text}`}>
                {fmtNum(conf, 1)}%
              </span>
            </div>
            <ProgressBar value={conf} tone={tone} className="mt-1.5" />
            <div className="mt-1 flex justify-between text-[10px] text-muted-2">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

// =====================================================================
//  2️⃣ TimeframeStrip —— 多周期 Tabs（横排 + badge + tooltip）
// =====================================================================

/**
 * 多周期共振条：5m / 15m / 1h / 4h / 1d 横向 chip
 * - 每个 chip 显示周期 + 当前 bias badge
 * - hover 显示周期解释（tooltip）
 * - 选中 tab 切换为 "全部" 时返回汇总
 */
function TimeframeStrip({ data }: { data: TimeframeAlignmentItem[] }) {
  const [active, setActive] = useState<string>('all')
  if (!data.length) return null

  /** 投票统计 */
  const counts = data.reduce(
    (acc, x) => {
      if (x.color === 'success') acc.long += 1
      else if (x.color === 'error') acc.short += 1
      else acc.neutral += 1
      return acc
    },
    { long: 0, short: 0, neutral: 0 },
  )

  const tfDescriptions: Record<string, string> = {
    '5m': '5 分钟级 · 噪声大，主要看微结构与瞬时流动性',
    '15m': '15 分钟级 · 短线节奏，触发与离场依据',
    '1h': '1 小时级 · 主交易级别，兼顾趋势与波段',
    '4h': '4 小时级 · 中线背景，决定方向偏置',
    '1d': '日线级 · 大方向锚点，规避逆势',
  }

  return (
    <Card
      eyebrow="MULTI-TIMEFRAME"
      title="多周期共振"
      description="跨周期方向投票。一致性越高，信号质量越好。"
      action={
        <div className="flex items-center gap-1.5 text-[11px]">
          {counts.long > 0 && (
            <Badge tone="long" variant="soft" size="xs">
              多 · {counts.long}
            </Badge>
          )}
          {counts.short > 0 && (
            <Badge tone="short" variant="soft" size="xs">
              空 · {counts.short}
            </Badge>
          )}
          {counts.neutral > 0 && (
            <Badge tone="neutral" variant="soft" size="xs">
              观望 · {counts.neutral}
            </Badge>
          )}
        </div>
      }
      delay={0.1}
    >
      {/* 周期 tabs */}
      <div className="flex flex-wrap gap-2">
        <TabButton
          active={active === 'all'}
          onClick={() => setActive('all')}
          label="全部"
        />
        {data.map((tf) => (
          <TabButton
            key={tf.timeframe}
            active={active === tf.timeframe}
            onClick={() => setActive(tf.timeframe)}
            label={tf.timeframe}
            tone={
              tf.color === 'success'
                ? 'long'
                : tf.color === 'error'
                  ? 'short'
                  : 'neutral'
            }
          />
        ))}
      </div>

      {/* 周期卡片网格 */}
      <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {data
          .filter((tf) => active === 'all' || active === tf.timeframe)
          .map((tf) => {
            const tone: Tone =
              tf.color === 'success'
                ? 'long'
                : tf.color === 'error'
                  ? 'short'
                  : 'neutral'
            const palette = biasPalette(tf.color)
            return (
              <Tooltip
                key={tf.timeframe}
                content={tfDescriptions[tf.timeframe] ?? '该周期方向投票'}
                className="block"
              >
                <div
                  className={`group flex h-full cursor-help flex-col gap-2 rounded-xl border p-4 transition hover:-translate-y-[1px] hover:shadow-card ${palette.border} ${palette.bgSoft}`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[11px] font-semibold uppercase tracking-wide text-ink-2">
                      {tf.timeframe}
                    </span>
                    <span
                      className={`inline-block h-1.5 w-1.5 rounded-full ${palette.bar}`}
                    />
                  </div>
                  <div className={`text-xl font-bold ${palette.text}`}>
                    {tf.label}
                  </div>
                  <Badge tone={tone} variant="solid" size="xs">
                    {tf.bias.toUpperCase()}
                  </Badge>
                </div>
              </Tooltip>
            )
          })}
      </div>
    </Card>
  )
}

/** Tab 按钮：active 时实心，inactive 时浅描边 */
function TabButton({
  active,
  onClick,
  label,
  tone = 'accent',
}: {
  active: boolean
  onClick: () => void
  label: string
  tone?: Tone
}) {
  const toneActive: Record<Tone, string> = {
    long: 'bg-long text-white border-long',
    short: 'bg-short text-white border-short',
    neutral: 'bg-neutral text-white border-neutral',
    accent: 'bg-ink text-white border-ink',
    muted: 'bg-ink text-white border-ink',
  }
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex h-8 items-center rounded-lg border px-3 text-xs font-medium transition ${
        active
          ? toneActive[tone]
          : 'border-hairline bg-surface text-ink-2 hover:border-hairline-strong hover:text-ink'
      }`}
    >
      {label}
    </button>
  )
}

// =====================================================================
//  3️⃣ TradingPlanCard —— 交易计划（入场区间 / 止损 / 止盈梯度）
// =====================================================================

/**
 * 交易计划卡片
 * - 三个关键指标横排：入场区间 / 止损 / 仓位与RR
 * - 价格刻度尺（visualization）：把 SL · entry · TP · LAST 标注在同一条轴上
 * - TP 列表：每个 TP 一个小卡片（价格 + 收益率）
 */
function TradingPlanCard({ data }: { data: AnalysisFull }) {
  const { trading_plan: plan, decision, summary } = data
  const entry = plan.entry_zone
  const tps = plan.take_profit
  const palette = biasPalette(decision.bias_color)

  if (!plan.has_plan || !entry) {
    return (
      <Card eyebrow="TRADING PLAN" title="本轮无明确交易计划" delay={0.18}>
        <div className="flex items-start gap-3 rounded-xl bg-bg-2 p-4">
          <svg
            width="20"
            height="20"
            viewBox="0 0 20 20"
            fill="none"
            className="mt-0.5 shrink-0 text-muted"
            aria-hidden
          >
            <circle cx="10" cy="10" r="8" stroke="currentColor" strokeWidth="1.4" />
            <path
              d="M10 6v4M10 13.5v.01"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          </svg>
          <p className="text-sm leading-relaxed text-ink-2">
            当前判断为
            <Badge tone="neutral" variant="soft" size="xs" className="mx-1">
              {decision.bias_label}
            </Badge>
            ，引擎未给出入场区间。当多周期共振或规则评分达到阈值后会自动生成。
          </p>
        </div>
      </Card>
    )
  }

  // 用 entry / stop / tp / last 的最小 / 最大价格确定刻度尺范围
  const allPrices = [
    entry.low,
    entry.high,
    plan.stop_loss ?? entry.low,
    ...tps.map((t) => t.price),
    summary.current_price ?? entry.low,
  ].filter((v): v is number => typeof v === 'number')
  const minP = Math.min(...allPrices)
  const maxP = Math.max(...allPrices)
  const span = Math.max(1e-9, maxP - minP)
  /** 把价格映射到 0~100 的横向百分比 */
  const pct = (p: number) => ((p - minP) / span) * 100

  return (
    <Card
      eyebrow="TRADING PLAN"
      title="交易计划"
      description="入场区间 / 止损 / 止盈梯度，按当前价定位"
      delay={0.18}
    >
      {/* 三段关键参数 */}
      <div className="grid grid-cols-3 gap-3 sm:gap-4">
        <div className="rounded-xl border border-hairline bg-surface-2 p-3.5">
          <div className="text-[11px] font-medium uppercase tracking-wider text-muted">
            入场区间
          </div>
          <div className="mt-1.5 font-display text-base font-semibold tabular text-ink sm:text-lg">
            ${fmtPrice(entry.low)}
            <span className="mx-1 text-muted">→</span>${fmtPrice(entry.high)}
          </div>
          <div className="mt-1 text-[11px] text-muted">
            中位 ${fmtPrice(entry.mid)} · 宽度{' '}
            {entry.width_pct ? (entry.width_pct * 100).toFixed(2) + '%' : '—'}
          </div>
        </div>
        <div className="rounded-xl border border-hairline bg-surface-2 p-3.5">
          <div className="text-[11px] font-medium uppercase tracking-wider text-muted">
            止损
          </div>
          <div className="mt-1.5 font-display text-base font-semibold tabular text-short sm:text-lg">
            ${fmtPrice(plan.stop_loss)}
          </div>
          <div className="mt-1 text-[11px] text-muted">
            风险 {fmtPct(plan.stop_loss_pct, 2)}
          </div>
        </div>
        <div className="rounded-xl border border-hairline bg-surface-2 p-3.5">
          <div className="text-[11px] font-medium uppercase tracking-wider text-muted">
            仓位 · RR
          </div>
          <div className="mt-1.5 font-display text-base font-semibold tabular text-ink sm:text-lg">
            {plan.position_size_label ?? '—'}
            <span className="mx-1.5 text-muted-2">·</span>
            <span>{fmtNum(plan.risk_reward_ratio ?? null, 2)}</span>
          </div>
          <div className="mt-1 text-[11px] text-muted">
            占资 · 风险收益比
          </div>
        </div>
      </div>

      {/* 价格刻度尺 */}
      <div className="mt-7 rounded-xl border border-hairline bg-surface-2 px-4 py-7">
        <div className="relative h-[60px]">
          {/* 底轴 */}
          <div className="absolute inset-x-0 top-1/2 h-px bg-hairline-strong" />
          {/* 入场区间块 */}
          <div
            className={`absolute top-1/2 h-3 -translate-y-1/2 rounded-sm ${palette.bg} opacity-30`}
            style={{
              left: `${pct(entry.low)}%`,
              width: `${pct(entry.high) - pct(entry.low)}%`,
            }}
          />
          <div
            className={`absolute top-1/2 h-3 -translate-y-1/2 border-y-2 ${palette.border}`}
            style={{
              left: `${pct(entry.low)}%`,
              width: `${pct(entry.high) - pct(entry.low)}%`,
            }}
          />
          {/* 止损 */}
          {plan.stop_loss !== null && (
            <Marker
              left={pct(plan.stop_loss)}
              label="SL"
              price={plan.stop_loss}
              tone="text-short"
            />
          )}
          {/* 止盈 */}
          {tps.map((tp, i) => (
            <Marker
              key={tp.level}
              left={pct(tp.price)}
              label={`TP${i + 1}`}
              price={tp.price}
              tone="text-long"
              upward
            />
          ))}
          {/* 当前价 */}
          {summary.current_price !== null &&
            summary.current_price !== undefined && (
              <Marker
                left={Math.max(0, Math.min(100, pct(summary.current_price)))}
                label="现价"
                price={summary.current_price}
                tone="text-accent"
                upward
                accent
              />
            )}
        </div>
      </div>

      {/* TP 网格 */}
      {tps.length > 0 && (
        <div className="mt-5 grid grid-cols-1 gap-2 sm:grid-cols-2">
          {tps.map((tp, i) => (
            <div
              key={tp.level}
              className="flex items-center justify-between rounded-xl border border-hairline bg-surface px-4 py-2.5 transition hover:border-long-soft hover:bg-long-bg"
            >
              <div className="flex items-center gap-2">
                <span className="grid h-6 w-10 place-items-center rounded-md bg-long-bg text-[11px] font-semibold text-long">
                  TP{i + 1}
                </span>
                <span className="font-mono text-sm tabular text-ink">
                  ${fmtPrice(tp.price)}
                </span>
              </div>
              <span
                className={`text-sm font-semibold tabular ${
                  (tp.reward_pct ?? 0) >= 0 ? 'text-long' : 'text-short'
                }`}
              >
                {fmtPct(tp.reward_pct, 2)}
              </span>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

/** 价格刻度尺上的小标记（垂直线 + 上下方文字） */
function Marker({
  left,
  label,
  price,
  tone,
  upward = false,
  accent = false,
}: {
  /** 在轴上的百分比位置（0-100） */
  left: number
  /** 顶部/底部小字 */
  label: string
  /** 价格（用于显示） */
  price: number
  /** 颜色 class */
  tone: string
  /** 是否向上展示文字（避免重叠） */
  upward?: boolean
  /** 是否使用 accent 色（一般给"现价"使用） */
  accent?: boolean
}) {
  return (
    <div
      className="absolute top-0 h-full"
      style={{ left: `${left}%`, transform: 'translateX(-50%)' }}
    >
      <div
        className={`absolute left-1/2 h-full w-px -translate-x-1/2 ${
          accent ? 'bg-accent' : 'bg-hairline-strong'
        }`}
      />
      <div
        className={`absolute left-1/2 ${
          upward ? 'bottom-1/2 mb-2' : 'top-1/2 mt-2'
        } -translate-x-1/2 whitespace-nowrap text-center font-mono text-[10px] tabular ${tone}`}
      >
        <div className="font-semibold">{label}</div>
        <div className="text-ink">${fmtPrice(price)}</div>
      </div>
    </div>
  )
}

// =====================================================================
//  LifecyclePanel —— 信号生命周期 / 实战表现
// =====================================================================

/**
 * 信号生命周期卡片
 * - 顶部：状态 chip + open/settled 标签
 * - 中部：PnL / MFE / MAE 三大数字
 * - 底部：触发 / 退出 / 失效时间表
 */
function LifecyclePanel({
  data,
  bias,
}: {
  data: AnalysisFull['lifecycle']
  bias: string
}) {
  const cls = lifecyclePalette(data.status_color)
  const pnl = data.pnl_pct
  const mfe = data.max_favorable_pct
  const mae = data.max_adverse_pct

  return (
    <Card
      eyebrow="LIFECYCLE"
      title="信号实战表现"
      description="自触发以来的盈亏与极值"
      action={
        <Badge tone="muted" variant="outline" size="xs">
          {data.is_open ? '进行中' : data.is_settled ? '已结算' : '空闲'}
        </Badge>
      }
      delay={0.22}
    >
      <div
        className={`flex items-center gap-2 rounded-xl border px-3 py-2 ${cls}`}
      >
        <span className="inline-block h-2 w-2 rounded-full bg-current" />
        <span className="text-sm font-semibold">{data.status_label}</span>
      </div>

      <div className="mt-5 grid grid-cols-3 gap-3">
        <Stat
          label="PnL"
          value={fmtPct(pnl, 2)}
          tone={
            (pnl ?? 0) > 0
              ? 'text-long'
              : (pnl ?? 0) < 0
                ? 'text-short'
                : 'text-ink'
          }
          hint={
            bias === 'long' ? '多单收益' : bias === 'short' ? '空单收益' : '收益'
          }
          size="md"
        />
        <Stat
          label="MFE"
          value={fmtPct(mfe, 2)}
          tone="text-long"
          hint={
            <Tooltip content="Max Favorable Excursion · 最大有利偏移">
              <span className="cursor-help underline decoration-dotted">
                极值收益
              </span>
            </Tooltip>
          }
          size="md"
        />
        <Stat
          label="MAE"
          value={fmtPct(mae, 2)}
          tone="text-short"
          hint={
            <Tooltip content="Max Adverse Excursion · 最大不利偏移">
              <span className="cursor-help underline decoration-dotted">
                极值回撤
              </span>
            </Tooltip>
          }
          size="md"
        />
      </div>

      <dl className="mt-5 space-y-2 border-t border-hairline pt-4 text-[12px]">
        <TimelineRow
          label="触发"
          time={fmtLocalTime(data.triggered_at)}
          extra={
            data.triggered_price !== null
              ? `@ $${fmtPrice(data.triggered_price)}`
              : undefined
          }
        />
        <TimelineRow
          label="退出"
          time={fmtLocalTime(data.exit_at)}
          extra={
            data.exit_price !== null ? `@ $${fmtPrice(data.exit_price)}` : undefined
          }
        />
        <TimelineRow label="失效" time={fmtLocalTime(data.expires_at)} />
      </dl>
    </Card>
  )
}

/** 生命周期时间表的一行 */
function TimelineRow({
  label,
  time,
  extra,
}: {
  label: string
  time: string
  extra?: string
}) {
  return (
    <div className="flex items-baseline justify-between">
      <dt className="text-muted">{label}</dt>
      <dd className="text-right font-mono text-ink-2">
        {time}
        {extra && <span className="ml-1.5 text-muted-2">{extra}</span>}
      </dd>
    </div>
  )
}

// =====================================================================
//  4️⃣ AttributionList —— 因子贡献（分组 + 排序 + 进度条）
// =====================================================================

/**
 * 因子贡献度归因列表
 * - 分组 tabs（全部 + 各 group）
 * - 排序按钮（影响力降序 / 升序 / 仅正贡献 / 仅负贡献）
 * - 进度条 + 正负分色
 */
function AttributionList({ data }: { data: AnalysisFull['rule_engine'] }) {
  type SortMode = 'impact' | 'positive' | 'negative'
  const [group, setGroup] = useState<string>('all')
  const [sort, setSort] = useState<SortMode>('impact')

  const items = data.top_contributions
  /** 所有可用 group */
  const groups = useMemo(() => {
    const s = new Set<string>()
    items.forEach((x) => x.group && s.add(x.group))
    return Array.from(s)
  }, [items])

  /** 经过过滤 + 排序 */
  const filtered = useMemo(() => {
    let r = items
    if (group !== 'all') r = r.filter((x) => x.group === group)
    if (sort === 'positive') r = r.filter((x) => x.contribution > 0)
    if (sort === 'negative') r = r.filter((x) => x.contribution < 0)
    return [...r].sort((a, b) => b.abs_contribution - a.abs_contribution)
  }, [items, group, sort])

  if (!items.length) {
    return (
      <Card eyebrow="ATTRIBUTION" title="因子贡献度归因" delay={0.26}>
        <p className="text-sm text-muted">本次未生成有效因子贡献。</p>
      </Card>
    )
  }
  const maxAbs = Math.max(...items.map((x) => x.abs_contribution), 1e-9)

  return (
    <Card
      eyebrow="ATTRIBUTION"
      title="因子贡献度归因"
      description="规则引擎给每条因子的边际贡献。颜色：绿=利多 · 红=利空"
      action={
        <Badge tone="muted" variant="soft" size="xs">
          总规则评分 ·{' '}
          <span
            className={`ml-1 font-semibold ${
              (data.rule_score ?? 0) > 0
                ? 'text-long'
                : (data.rule_score ?? 0) < 0
                  ? 'text-short'
                  : 'text-ink'
            }`}
          >
            {fmtNum(data.rule_score, 3)}
          </span>
        </Badge>
      }
      delay={0.26}
    >
      {/* 顶部过滤 */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1.5">
          <TabButton
            active={group === 'all'}
            onClick={() => setGroup('all')}
            label="全部"
          />
          {groups.map((g) => (
            <TabButton
              key={g}
              active={group === g}
              onClick={() => setGroup(g)}
              label={g}
            />
          ))}
        </div>

        <div className="inline-flex items-center rounded-lg border border-hairline bg-surface p-0.5">
          <SortChip active={sort === 'impact'} onClick={() => setSort('impact')}>
            影响力
          </SortChip>
          <SortChip
            active={sort === 'positive'}
            onClick={() => setSort('positive')}
            tone="long"
          >
            仅利多
          </SortChip>
          <SortChip
            active={sort === 'negative'}
            onClick={() => setSort('negative')}
            tone="short"
          >
            仅利空
          </SortChip>
        </div>
      </div>

      {/* 列表 */}
      <ul className="divide-y divide-hairline">
        {filtered.length === 0 && (
          <li className="py-6 text-center text-sm text-muted">
            没有匹配的因子。
          </li>
        )}
        {filtered.slice(0, 12).map((it) => {
          const positive = it.contribution >= 0
          const widthPct = (it.abs_contribution / maxAbs) * 100
          return (
            <li
              key={it.key}
              className="group grid grid-cols-12 items-center gap-3 py-2.5"
            >
              <div className="col-span-4 flex items-center gap-2 truncate sm:col-span-3">
                <Badge tone="muted" variant="outline" size="xs">
                  {it.timeframe || '—'}
                </Badge>
                {it.group && (
                  <Badge tone="accent" variant="soft" size="xs">
                    {it.group}
                  </Badge>
                )}
              </div>
              <div className="col-span-5 truncate text-sm font-medium text-ink sm:col-span-5">
                <Tooltip
                  content={`贡献 ${it.contribution >= 0 ? '+' : ''}${it.contribution.toFixed(4)}`}
                >
                  <span className="cursor-help">{it.factor}</span>
                </Tooltip>
              </div>
              <div className="col-span-3 sm:col-span-3">
                <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-bg-2">
                  <div
                    className={`absolute inset-y-0 left-0 rounded-full transition-all duration-700 ${
                      positive ? 'bg-long' : 'bg-short'
                    } group-hover:opacity-80`}
                    style={{ width: `${widthPct}%` }}
                  />
                </div>
              </div>
              <div
                className={`col-span-12 mt-0.5 text-right font-mono text-[12px] font-semibold tabular sm:col-span-1 sm:mt-0 ${
                  positive ? 'text-long' : 'text-short'
                }`}
              >
                {positive ? '+' : ''}
                {it.contribution.toFixed(3)}
              </div>
            </li>
          )
        })}
      </ul>
    </Card>
  )
}

/** AttributionList 顶部的排序 chip（小型） */
function SortChip({
  active,
  onClick,
  children,
  tone = 'accent',
}: {
  active: boolean
  onClick: () => void
  children: ReactNode
  tone?: 'accent' | 'long' | 'short'
}) {
  const activeTone: Record<typeof tone, string> = {
    accent: 'bg-bg-2 text-ink',
    long: 'bg-long-bg text-long',
    short: 'bg-short-bg text-short',
  }
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-md px-2 py-1 text-[11px] font-medium transition ${
        active ? activeTone[tone] : 'text-muted hover:text-ink'
      }`}
    >
      {children}
    </button>
  )
}

// =====================================================================
//  MarketContextCard —— 市场结构（流动性 / MTF / Regime）
// =====================================================================

/**
 * 市场环境卡片：
 * - 上：上下流动性池占比（柱状）
 * - 中：MTF 分数 + 主导方向
 * - 下：regime
 */
function MarketContextCard({
  liquidity,
  mtf,
  regime,
}: {
  liquidity: AnalysisFull['market_context']['liquidity']
  mtf: AnalysisFull['market_context']['mtf_alignment']
  regime: string | null
}) {
  const above = liquidity?.pool_above_count ?? 0
  const below = liquidity?.pool_below_count ?? 0
  const total = Math.max(1, above + below)
  /** mtf 分数转成 0-100，绝对值越大共振越强 */
  const mtfScore = mtf?.alignment_score ?? null
  const mtfPct = mtfScore === null ? 0 : Math.min(100, Math.abs(mtfScore) * 100)
  const dominantTone: Tone =
    mtf?.dominant_bias === 'long'
      ? 'long'
      : mtf?.dominant_bias === 'short'
        ? 'short'
        : 'neutral'

  return (
    <Card
      eyebrow="MARKET CONTEXT"
      title="市场结构"
      description="流动性 · 共振 · 市场状态"
      delay={0.3}
    >
      {/* 上下流动性池可视化 */}
      <div className="rounded-xl border border-hairline bg-surface-2 p-4">
        <div className="text-[11px] font-medium uppercase tracking-wider text-muted">
          上下流动性池
        </div>
        <div className="mt-3 space-y-3">
          <div className="flex items-center gap-3">
            <Tooltip content="上方流动性池数量。距离越近越易被扫">
              <span className="w-14 cursor-help text-[11px] font-medium text-ink-2">
                上方 {fmtPct(liquidity?.nearest_above_pct ?? null, 2)}
              </span>
            </Tooltip>
            <div className="flex-1">
              <ProgressBar
                value={(above / total) * 100}
                tone="long"
                height="h-2.5"
              />
            </div>
            <span className="w-8 text-right font-mono text-xs font-semibold tabular text-long">
              {above}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <Tooltip content="下方流动性池数量">
              <span className="w-14 cursor-help text-[11px] font-medium text-ink-2">
                下方 {fmtPct(liquidity?.nearest_below_pct ?? null, 2)}
              </span>
            </Tooltip>
            <div className="flex-1">
              <ProgressBar
                value={(below / total) * 100}
                tone="short"
                height="h-2.5"
              />
            </div>
            <span className="w-8 text-right font-mono text-xs font-semibold tabular text-short">
              {below}
            </span>
          </div>
        </div>
      </div>

      {/* MTF + Regime 双块 */}
      <div className="mt-3 grid grid-cols-2 gap-3">
        <div className="rounded-xl border border-hairline bg-surface-2 p-4">
          <div className="text-[11px] font-medium uppercase tracking-wider text-muted">
            MTF 共振
          </div>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="font-display text-xl font-semibold tabular text-ink">
              {fmtNum(mtfScore, 3)}
            </span>
            <Badge tone={dominantTone} variant="soft" size="xs">
              {mtf?.dominant_bias ?? '—'}
            </Badge>
          </div>
          <ProgressBar
            value={mtfPct}
            tone={dominantTone}
            className="mt-3"
            height="h-1.5"
          />
        </div>

        <div className="rounded-xl border border-hairline bg-surface-2 p-4">
          <div className="text-[11px] font-medium uppercase tracking-wider text-muted">
            市场状态
          </div>
          <div className="mt-1 truncate text-base font-semibold text-ink">
            {regime ?? '—'}
          </div>
          <div className="mt-3 flex items-center gap-1.5">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
            <span className="text-[11px] text-muted">实时识别</span>
          </div>
        </div>
      </div>
    </Card>
  )
}

// =====================================================================
//  5️⃣ NarrativeCards —— Reasoning / Risk / Suggestion 三段卡片
// =====================================================================

/**
 * AI 解释三段式
 * - 判断依据（accent 蓝色）
 * - 风险提示（红色强调）
 * - 操作建议（绿色强调）
 * 每块限制为折叠状态，超出后"展开查看更多"
 */
function NarrativeCards({
  decision,
}: {
  decision: AnalysisFull['decision']
}) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      <NarrativeCard
        eyebrow="判断依据"
        title="为什么这么判断"
        body={decision.reason}
        tone="accent"
        icon={
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <path
              d="M9 12V8M9 5.5v.01"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
            <circle cx="9" cy="9" r="7" stroke="currentColor" strokeWidth="1.4" />
          </svg>
        }
        delay={0.34}
      />
      <NarrativeCard
        eyebrow="风险提示"
        title="可能出错在哪"
        body={decision.risk}
        tone="short"
        icon={
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <path
              d="M9 2 17 16H1L9 2Z"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinejoin="round"
            />
            <path
              d="M9 8v3M9 13.5v.01"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
          </svg>
        }
        delay={0.38}
      />
      <NarrativeCard
        eyebrow="操作建议"
        title="接下来怎么做"
        body={decision.suggestion}
        tone="long"
        icon={
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <path
              d="m4 9 3.5 3.5L14 6"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        }
        delay={0.42}
      />
    </div>
  )
}

/** Narrative 单个卡片：左侧色条 + 图标 + 标题 + 正文（可折叠） */
function NarrativeCard({
  eyebrow,
  title,
  body,
  tone,
  icon,
  delay,
}: {
  eyebrow: string
  title: string
  body: string
  tone: Tone
  icon: ReactNode
  delay: number
}) {
  const palette: Record<Tone, { ring: string; bg: string; text: string }> = {
    long: { ring: 'before:bg-long', bg: 'bg-long-bg', text: 'text-long' },
    short: { ring: 'before:bg-short', bg: 'bg-short-bg', text: 'text-short' },
    neutral: { ring: 'before:bg-neutral', bg: 'bg-neutral-bg', text: 'text-neutral' },
    accent: { ring: 'before:bg-accent', bg: 'bg-accent-bg', text: 'text-accent' },
    muted: { ring: 'before:bg-muted', bg: 'bg-bg-2', text: 'text-ink' },
  }
  const p = palette[tone]
  const text = body || '—'
  /** 是否需要折叠（粗略按字符数） */
  const longText = text.length > 140

  return (
    <article
      className={`rise-in relative overflow-hidden rounded-2xl border border-hairline bg-surface p-5 shadow-card transition hover:-translate-y-[1px] hover:shadow-card-hover before:absolute before:inset-y-4 before:left-0 before:w-[3px] before:rounded-r-full before:content-[''] ${p.ring}`}
      style={{ animationDelay: `${delay}s` }}
    >
      <div className="flex items-center gap-2">
        <span
          className={`grid h-7 w-7 place-items-center rounded-lg ${p.bg} ${p.text}`}
        >
          {icon}
        </span>
        <span className="text-[11px] font-semibold uppercase tracking-wider text-muted">
          {eyebrow}
        </span>
      </div>
      <h4 className={`mt-3 text-base font-semibold tracking-tight ${p.text}`}>
        {title}
      </h4>
      {longText ? (
        <Collapsible collapsedHeight={88} expandLabel="展开查看更多" collapseLabel="收起">
          <p className="mt-2 text-[13.5px] leading-[1.7] text-ink-2">{text}</p>
        </Collapsible>
      ) : (
        <p className="mt-2 text-[13.5px] leading-[1.7] text-ink-2">{text}</p>
      )}
    </article>
  )
}

// =====================================================================
//  6️⃣ HistoryTimeline —— 历史信号时间轴（横向滚动）
// =====================================================================

/**
 * 横向滚动的历史信号卡片
 * - hover 高亮 + 浮起
 * - 选中态环形 ring
 * - 每个卡片：时间 / 判词 / 现价 / RR / PnL / 状态
 */
function HistoryTimeline({
  items,
  activeId,
  onSelect,
}: {
  items: AnalysisFull[]
  activeId: number | null
  onSelect: (id: number) => void
}) {
  return (
    <Card
      eyebrow="HISTORY"
      title="历史信号时间轴"
      description="点击任一条切换为该次分析详情"
      action={
        <Badge tone="muted" variant="outline" size="xs">
          最近 {items.length} 条
        </Badge>
      }
      delay={0.46}
    >
      <div className="-mx-2 overflow-x-auto pb-2">
        <div className="flex min-w-full gap-3 px-2">
          {items.length === 0 && (
            <div className="py-6 text-sm text-muted">暂无历史。</div>
          )}
          {items.map((it) => {
            const palette = biasPalette(it.summary.bias_color)
            const tone: Tone =
              it.summary.bias_color === 'success'
                ? 'long'
                : it.summary.bias_color === 'error'
                  ? 'short'
                  : 'neutral'
            const active = it.signal_id === activeId
            const pnl = it.summary.pnl_pct
            return (
              <button
                key={it.signal_id ?? Math.random()}
                onClick={() => it.signal_id && onSelect(it.signal_id)}
                className={`group relative flex w-[208px] shrink-0 flex-col items-stretch gap-2 rounded-xl border bg-surface p-4 text-left transition hover:-translate-y-[1px] hover:shadow-card-hover ${
                  active
                    ? 'border-accent ring-2 ring-accent-soft'
                    : 'border-hairline hover:border-hairline-strong'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[11px] text-muted">
                    {fmtLocalTime(it.ts)}
                  </span>
                  <span
                    className={`inline-block h-1.5 w-1.5 rounded-full ${palette.bar}`}
                  />
                </div>
                <div className={`text-2xl font-bold ${palette.text}`}>
                  {it.summary.bias_label}
                </div>
                <div className="font-mono text-[12px] tabular text-ink-2">
                  ${fmtPrice(it.summary.current_price)}
                </div>
                <div className="flex items-center gap-1.5">
                  <Badge tone="muted" variant="outline" size="xs">
                    RR {fmtNum(it.summary.risk_reward_ratio ?? null, 2)}
                  </Badge>
                  <Badge
                    tone={
                      (pnl ?? 0) > 0 ? 'long' : (pnl ?? 0) < 0 ? 'short' : 'muted'
                    }
                    variant="soft"
                    size="xs"
                  >
                    {fmtPct(pnl, 1)}
                  </Badge>
                </div>
                <div className="mt-1 flex items-center gap-1.5">
                  <Badge tone={tone} variant="soft" size="xs">
                    {it.summary.lifecycle_status_label}
                  </Badge>
                </div>
              </button>
            )
          })}
        </div>
      </div>
    </Card>
  )
}

// =====================================================================
//  ReasoningTrigger / ReasoningDrawer —— 思维链全文阅读
// =====================================================================

/** 浮动按钮（右下角） */
function ReasoningTrigger({
  available,
  chars,
  onOpen,
}: {
  available: boolean
  chars: number
  onOpen: () => void
}) {
  return (
    <div className="fixed bottom-6 right-6 z-40">
      <button
        type="button"
        disabled={!available}
        onClick={onOpen}
        className="group flex items-center gap-3 rounded-full border border-hairline bg-surface px-4 py-2.5 shadow-card-hover transition hover:border-accent hover:bg-accent-bg disabled:cursor-not-allowed disabled:opacity-40"
      >
        <span className="grid h-7 w-7 place-items-center rounded-full bg-accent-bg text-accent">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
            <path
              d="M2 4h10M2 7h10M2 10h6"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        </span>
        <div className="text-left leading-tight">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted">
            CHAIN OF THOUGHT
          </div>
          <div className="text-xs font-medium text-ink">
            {available ? `阅读思维链 · ${chars.toLocaleString()} 字` : '本次无思维链'}
          </div>
        </div>
      </button>
    </div>
  )
}

/** 思维链右侧抽屉 */
function ReasoningDrawer({
  open,
  onClose,
  content,
  available,
  totalChars,
}: {
  open: boolean
  onClose: () => void
  content: string | null
  available: boolean
  totalChars: number
}) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <button
        aria-label="close"
        onClick={onClose}
        className="absolute inset-0 bg-ink/30 backdrop-blur-sm"
      />
      <aside className="drawer-in relative flex h-full w-full max-w-[680px] flex-col bg-surface shadow-modal">
        <div className="flex items-start justify-between border-b border-hairline px-6 py-5">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wider text-muted">
              CHAIN OF THOUGHT
            </div>
            <h3 className="mt-1 text-xl font-semibold tracking-tight text-ink">
              引擎思维链
            </h3>
            <p className="mt-1 text-[12px] text-muted">
              {totalChars.toLocaleString()} 字 · DeepSeek reasoning_content
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg border border-hairline bg-surface p-2 text-muted transition hover:border-hairline-strong hover:text-ink"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
              <path
                d="M2 2l10 10M12 2L2 12"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-6 py-6">
          {!available ? (
            <p className="text-sm text-muted">
              本次未启用思考模式（rules-only 路径）。
            </p>
          ) : content ? (
            <pre className="whitespace-pre-wrap break-words font-cn text-[14px] leading-[1.95] text-ink-2">
              {content}
            </pre>
          ) : (
            <div className="space-y-3">
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-3 w-11/12" />
              <Skeleton className="h-3 w-3/4" />
              <Skeleton className="h-3 w-10/12" />
            </div>
          )}
        </div>
      </aside>
    </div>
  )
}

// =====================================================================
//  Footer
// =====================================================================
function Footer() {
  return (
    <footer className="border-t border-hairline bg-surface">
      <div className="mx-auto flex max-w-[1320px] flex-wrap items-center justify-between gap-3 px-4 py-6 text-[11px] text-muted sm:px-6 lg:px-10">
        <div>Alpha · ETH 量化分析工作台 · 仅供参考，不构成投资建议</div>
        <div className="font-mono">
          Rule Engine + DeepSeek-V4 · Multi-Timeframe Resonance
        </div>
      </div>
    </footer>
  )
}
