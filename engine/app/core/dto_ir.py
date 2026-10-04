"""
TrustLens v2 DTO IR — canonical Pydantic schema.

本版本将 DTO IR 拆分为两个独立顶层类型：
  - ReconciliationDTOIR：供 ReconciliationEngine 消费（document + reconciliation）
  - GroundingDTOIR：     供 GroundingEngine 消费（grounding targets only）

Grounding 目标合并为单一 `targets` 列表（不再分 web / enterprise）。
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================
# Primitive
# ============================================================

DecimalStr = Annotated[
    str,
    Field(pattern=r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$"),
]

CurrencyCode = Annotated[
    str,
    Field(pattern=r"^[A-Z]{3}$"),
]


# ============================================================
# Source
# ============================================================

class SourceRef(BaseModel):
    """
    回指物理层。只保留 observation_ids。

    observation_id 语义（由 Perception 层保证）：
      id = page * 1000 + local_idx
      page = id // 1000          （1-indexed）
      local_idx = id % 1000      （页内位置，0-indexed）
    """
    model_config = ConfigDict(extra="forbid")

    observation_ids: list[int] = Field(default_factory=list)


# ============================================================
# Values
# ============================================================

class CurrencyValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: DecimalStr
    currency: CurrencyCode


class PercentageValue(BaseModel):
    """Percentage expressed as a number, e.g. "11" means 11%, not 0.11."""
    model_config = ConfigDict(extra="forbid")
    value: DecimalStr
    unit: Literal["PERCENT"] = "PERCENT"


class QuantityValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: DecimalStr
    unit: str | None = None


# ============================================================
# Document
# ============================================================

class DocumentType(str, Enum):
    PAYSLIP = "PAYSLIP"
    BANK_STATEMENT = "BANK_STATEMENT"
    INVOICE = "INVOICE"
    QUOTATION = "QUOTATION"
    RECEIPT = "RECEIPT"
    E_RECEIPT = "E_RECEIPT"
    CERTIFICATE = "CERTIFICATE"
    RESUME = "RESUME"
    OFFICIAL_DOC = "OFFICIAL_DOC"
    LEGAL_DOC = "LEGAL_DOC"


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    document_type: DocumentType
    page_count: int | None = Field(default=None, ge=1)
    source: SourceRef | None = None


# ============================================================
# Global Facts
# ============================================================

class GlobalFactRole(str, Enum):
    OPENING_BALANCE = "OPENING_BALANCE"
    CLOSING_BALANCE = "CLOSING_BALANCE"

    SUBTOTAL = "SUBTOTAL"
    DISCOUNT_AMOUNT = "DISCOUNT_AMOUNT"
    TAX_AMOUNT = "TAX_AMOUNT"
    SHIPPING_FEE = "SHIPPING_FEE"
    TOTAL_AMOUNT = "TOTAL_AMOUNT"

    AMOUNT_PAID = "AMOUNT_PAID"
    AMOUNT_DUE = "AMOUNT_DUE"

    GROSS_PAY = "GROSS_PAY"
    NET_PAY = "NET_PAY"
    EMPLOYEE_DEDUCTION_TOTAL = "EMPLOYEE_DEDUCTION_TOTAL"
    EMPLOYER_CONTRIBUTION_TOTAL = "EMPLOYER_CONTRIBUTION_TOTAL"

    PRINCIPAL = "PRINCIPAL"
    INTEREST = "INTEREST"
    PENALTY = "PENALTY"
    COMPENSATION = "COMPENSATION"

    QUANTITY = "QUANTITY"
    HOURS = "HOURS"
    DAYS = "DAYS"
    MONTHS = "MONTHS"
    YEARS = "YEARS"

    TAX_RATE = "TAX_RATE"
    DISCOUNT_RATE = "DISCOUNT_RATE"
    INTEREST_RATE = "INTEREST_RATE"
    PENALTY_RATE = "PENALTY_RATE"

    ISSUE_DATE = "ISSUE_DATE"
    DUE_DATE = "DUE_DATE"
    PERIOD_START = "PERIOD_START"
    PERIOD_END = "PERIOD_END"
    VALID_FROM = "VALID_FROM"
    VALID_UNTIL = "VALID_UNTIL"
    DEADLINE = "DEADLINE"

    TRANSACTION_DATETIME = "TRANSACTION_DATETIME"
    PAYMENT_DATETIME = "PAYMENT_DATETIME"


GlobalFactValue = Union[
    CurrencyValue,
    PercentageValue,
    QuantityValue,
    date,
    datetime,
    DecimalStr,
]


class GlobalFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: GlobalFactRole
    value: GlobalFactValue
    source: SourceRef | None = None


# ============================================================
# Components (closed-world enums)
# ============================================================

class PayrollComponent(str, Enum):
    BASIC_SALARY = "BASIC_SALARY"
    OVERTIME_PAY = "OVERTIME_PAY"
    ALLOWANCE = "ALLOWANCE"
    BONUS = "BONUS"
    COMMISSION = "COMMISSION"
    OTHER_EARNING = "OTHER_EARNING"

    EPF_EMPLOYEE = "EPF_EMPLOYEE"
    SOCSO_EMPLOYEE = "SOCSO_EMPLOYEE"
    EIS_EMPLOYEE = "EIS_EMPLOYEE"
    TAX_DEDUCTION = "TAX_DEDUCTION"
    EMPLOYEE_DEDUCTION = "EMPLOYEE_DEDUCTION"
    OTHER_DEDUCTION = "OTHER_DEDUCTION"

    EPF_EMPLOYER = "EPF_EMPLOYER"
    SOCSO_EMPLOYER = "SOCSO_EMPLOYER"
    EIS_EMPLOYER = "EIS_EMPLOYER"
    EMPLOYER_CONTRIBUTION = "EMPLOYER_CONTRIBUTION"
    OTHER_CONTRIBUTION = "OTHER_CONTRIBUTION"


class LegalComponent(str, Enum):
    PRINCIPAL = "PRINCIPAL"
    INTEREST = "INTEREST"
    PENALTY = "PENALTY"
    LEGAL_FEE = "LEGAL_FEE"
    COMPENSATION = "COMPENSATION"
    AWARD = "AWARD"
    CLAIM_AMOUNT = "CLAIM_AMOUNT"
    LIABILITY_AMOUNT = "LIABILITY_AMOUNT"


class OfficialComponent(str, Enum):
    TAX = "TAX"
    DUTY = "DUTY"
    FEE = "FEE"
    PENALTY = "PENALTY"
    GRANT = "GRANT"
    BENEFIT = "BENEFIT"
    REFUND = "REFUND"
    ASSESSMENT = "ASSESSMENT"
    LIABILITY = "LIABILITY"
    AMOUNT_DUE = "AMOUNT_DUE"
    AMOUNT_PAID = "AMOUNT_PAID"


# ============================================================
# Matrix base
# ============================================================

Cell = str | None


class MatrixBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tuples: list[list[Cell]] = Field(default_factory=list)
    raw_headers: list[str] | None = None
    source_ids: list[list[int | list[int] | None]] | None = None

    @model_validator(mode="after")
    def _validate_matrix_shape(self):
        cols = getattr(self, "columns", None)
        if not cols:
            return self
        expected = len(cols)

        for i, row in enumerate(self.tuples):
            if len(row) != expected:
                raise ValueError(
                    f"Row {i} has {len(row)} cells, expected {expected}"
                )

        if self.source_ids is not None:
            if len(self.source_ids) != len(self.tuples):
                raise ValueError(
                    f"source_ids has {len(self.source_ids)} rows, "
                    f"expected {len(self.tuples)}"
                )
            for i, row in enumerate(self.source_ids):
                if len(row) != expected:
                    raise ValueError(
                        f"source_ids row {i} has {len(row)} cells, "
                        f"expected {expected}"
                    )
        return self

    def collect_all_obs_ids(self) -> list[int]:
        """把 source_ids 里所有 observation_id 收集起来，去重排序。"""
        if not self.source_ids:
            return []
        ids: set[int] = set()
        for row in self.source_ids:
            for cell in row:
                if isinstance(cell, list):
                    ids.update(i for i in cell if isinstance(i, int))
                elif isinstance(cell, int):
                    ids.add(cell)
        return sorted(ids)


# ============================================================
# Column enums
# ============================================================

class BankColumn(str, Enum):
    EVENT_DATE = "EVENT_DATE"
    POSTING_DATE = "POSTING_DATE"
    DESC = "DESC"
    REFERENCE = "REFERENCE"
    FLOW_IN = "FLOW_IN"
    FLOW_OUT = "FLOW_OUT"
    FLOW_SIGNED = "FLOW_SIGNED"
    RUNNING_BALANCE = "RUNNING_BALANCE"


class PayrollColumn(str, Enum):
    COMPONENT = "COMPONENT"
    AMOUNT = "AMOUNT"
    RATE = "RATE"
    QUANTITY = "QUANTITY"
    EVENT_DATE = "EVENT_DATE"


class CommercialColumn(str, Enum):
    PRODUCT = "PRODUCT"
    PRODUCT_CODE = "PRODUCT_CODE"
    QUANTITY = "QUANTITY"
    UNIT = "UNIT"
    UNIT_PRICE = "UNIT_PRICE"
    DISCOUNT_AMOUNT = "DISCOUNT_AMOUNT"
    DISCOUNT_RATE = "DISCOUNT_RATE"
    ROW_TAX = "ROW_TAX"
    TAX_RATE = "TAX_RATE"
    ROW_TOTAL = "ROW_TOTAL"


class EmploymentColumn(str, Enum):
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"


class EducationColumn(str, Enum):
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"


class CertificateColumn(str, Enum):
    ISSUE_DATE = "ISSUE_DATE"
    EXPIRY_DATE = "EXPIRY_DATE"
    VALID_FROM = "VALID_FROM"
    VALID_UNTIL = "VALID_UNTIL"


class LegalColumn(str, Enum):
    COMPONENT = "COMPONENT"
    AMOUNT = "AMOUNT"
    RATE = "RATE"
    EVENT_DATE = "EVENT_DATE"
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    DEADLINE = "DEADLINE"


class OfficialColumn(str, Enum):
    COMPONENT = "COMPONENT"
    AMOUNT = "AMOUNT"
    RATE = "RATE"
    EVENT_DATE = "EVENT_DATE"
    START_DATE = "START_DATE"
    END_DATE = "END_DATE"
    DEADLINE = "DEADLINE"


# ============================================================
# 8 Concrete tables
# ============================================================

class BankTransactionTable(MatrixBase):
    model_config = ConfigDict(extra="forbid")
    id: str
    table_type: Literal["BANK_TRANSACTIONS"] = "BANK_TRANSACTIONS"
    columns: list[BankColumn]


class PayrollTable(MatrixBase):
    model_config = ConfigDict(extra="forbid")
    id: str
    table_type: Literal["PAYROLL_COMPONENTS"] = "PAYROLL_COMPONENTS"
    columns: list[PayrollColumn]


class CommercialLinesTable(MatrixBase):
    model_config = ConfigDict(extra="forbid")
    id: str
    table_type: Literal["COMMERCIAL_LINES"] = "COMMERCIAL_LINES"
    columns: list[CommercialColumn]


class EmploymentTable(MatrixBase):
    model_config = ConfigDict(extra="forbid")
    id: str
    table_type: Literal["EMPLOYMENT"] = "EMPLOYMENT"
    columns: list[EmploymentColumn]


class EducationTable(MatrixBase):
    model_config = ConfigDict(extra="forbid")
    id: str
    table_type: Literal["EDUCATION"] = "EDUCATION"
    columns: list[EducationColumn]


class CertificateValidityTable(MatrixBase):
    model_config = ConfigDict(extra="forbid")
    id: str
    table_type: Literal["CERTIFICATE_VALIDITY"] = "CERTIFICATE_VALIDITY"
    columns: list[CertificateColumn]


class LegalAmountsTable(MatrixBase):
    model_config = ConfigDict(extra="forbid")
    id: str
    table_type: Literal["LEGAL_AMOUNTS"] = "LEGAL_AMOUNTS"
    columns: list[LegalColumn]


class OfficialAmountsTable(MatrixBase):
    model_config = ConfigDict(extra="forbid")
    id: str
    table_type: Literal["OFFICIAL_AMOUNTS"] = "OFFICIAL_AMOUNTS"
    columns: list[OfficialColumn]


ReconciliationTable = Union[
    BankTransactionTable,
    PayrollTable,
    CommercialLinesTable,
    EmploymentTable,
    EducationTable,
    CertificateValidityTable,
    LegalAmountsTable,
    OfficialAmountsTable,
]


# ============================================================
# Reconciliation
# ============================================================

class ReconciliationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    global_facts: list[GlobalFact] = Field(default_factory=list)
    tables: list[ReconciliationTable] = Field(default_factory=list)


# ============================================================
# Entity (unified)
# ============================================================

class EntityType(str, Enum):
    """
    Grounding 目标的闭世界实体类型。

    命名沿用旧 EnterpriseEntityType，并新增若干外部可验证类型。
    """
    # --- 原 EnterpriseEntityType 保留 ---
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    PRODUCT = "PRODUCT"
    EMPLOYEE = "EMPLOYEE"
    CUSTOMER = "CUSTOMER"
    VENDOR = "VENDOR"
    ACCOUNT = "ACCOUNT"
    BANK = "BANK"
    TRANSACTION = "TRANSACTION"
    INVOICE = "INVOICE"
    QUOTATION = "QUOTATION"
    RECEIPT = "RECEIPT"
    PURCHASE_ORDER = "PURCHASE_ORDER"
    CONTRACT = "CONTRACT"
    CERTIFICATE = "CERTIFICATE"
    CASE = "CASE"
    OTHER = "OTHER"
    # --- 新增 ---
    GOVERNMENT_AGENCY = "GOVERNMENT_AGENCY"
    UNIVERSITY = "UNIVERSITY"
    PROFESSIONAL_BODY = "PROFESSIONAL_BODY"
    LAW_FIRM = "LAW_FIRM"
    WEBSITE = "WEBSITE"
    ADDRESS = "ADDRESS"


class EnterpriseKeyType(str, Enum):
    PERSON_ID = "PERSON_ID"
    EMPLOYEE_ID = "EMPLOYEE_ID"
    CUSTOMER_ID = "CUSTOMER_ID"
    VENDOR_ID = "VENDOR_ID"
    ORGANIZATION_ID = "ORGANIZATION_ID"
    COMPANY_REGISTRATION_NO = "COMPANY_REGISTRATION_NO"

    PRODUCT_ID = "PRODUCT_ID"
    PRODUCT_CODE = "PRODUCT_CODE"
    SKU = "SKU"

    ACCOUNT_ID = "ACCOUNT_ID"
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    BANK_ID = "BANK_ID"

    TRANSACTION_ID = "TRANSACTION_ID"
    TRANSACTION_REFERENCE = "TRANSACTION_REFERENCE"

    INVOICE_ID = "INVOICE_ID"
    INVOICE_NUMBER = "INVOICE_NUMBER"

    QUOTATION_ID = "QUOTATION_ID"
    QUOTATION_NUMBER = "QUOTATION_NUMBER"

    RECEIPT_ID = "RECEIPT_ID"
    RECEIPT_NUMBER = "RECEIPT_NUMBER"

    PURCHASE_ORDER_ID = "PURCHASE_ORDER_ID"
    PURCHASE_ORDER_NUMBER = "PURCHASE_ORDER_NUMBER"

    CONTRACT_ID = "CONTRACT_ID"
    CONTRACT_NUMBER = "CONTRACT_NUMBER"

    CERTIFICATE_ID = "CERTIFICATE_ID"
    CERTIFICATE_NUMBER = "CERTIFICATE_NUMBER"

    CASE_ID = "CASE_ID"
    CASE_NUMBER = "CASE_NUMBER"

    OTHER_ID = "OTHER_ID"


class EnterpriseKey(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: EnterpriseKeyType
    value: str
    normalized_value: str | None = None


# ============================================================
# Grounding
# ============================================================

class GroundingTarget(BaseModel):
    """
    单个待验证实体。

    不携带 verification_intent——由 GroundingEngine 的确定性 router 决策。
    """
    model_config = ConfigDict(extra="forbid")

    entity_type: EntityType
    value: str
    keys: list[EnterpriseKey] = Field(default_factory=list)
    subkey: str | None = None
    source: SourceRef | None = None


class GroundingTargets(BaseModel):
    model_config = ConfigDict(extra="forbid")
    targets: list[GroundingTarget] = Field(default_factory=list)


# ============================================================
# Conflicts
# ============================================================

class DTOIRConflictType(str, Enum):
    INVALID_OBSERVATION_ID = "invalid_observation_id"
    ENUM_NORMALIZATION_FAILED = "enum_normalization_failed"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    TABLE_TYPE_MISMATCH = "table_type_mismatch"
    MISSING_SOURCE = "missing_source"
    DOCUMENT_TYPE_UNCERTAIN = "document_type_uncertain"
    VLM_JSON_PARSE_FAILED = "vlm_json_parse_failed"
    VLM_OCR_TEXT_MISMATCH = "vlm_ocr_text_mismatch"
    VLM_OCR_NUMERIC_MISMATCH = "vlm_ocr_numeric_mismatch"
    VLM_OCR_DATE_MISMATCH = "vlm_ocr_date_mismatch"
    VLM_DATE_FORMAT_VIOLATION = "vlm_date_format_violation"
    VLM_CALL_FAILED = "vlm_call_failed"
    OTHER = "other"


class DTOIRConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Literal["warning", "error"] = "warning"
    type: DTOIRConflictType
    message: str
    context: dict = Field(default_factory=dict)


# ============================================================
# DTO IR types
# ============================================================

class DTOIRBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conflicts: list[DTOIRConflict] = Field(default_factory=list)


class ReconciliationDTOIR(DTOIRBase):
    """
    IR1：供 ReconciliationEngine 消费。

    grounding 字段为**可选注入**：
      - VLM 的 reconciliation channel 不输出此字段
      - 调度层可将 GroundingDTOIR.grounding 注入此处，供 universal ID 校验
        （Luhn / MyKad）以及 invoice 的内嵌日期校验使用
      - 若为 None，依赖 grounding 的规则会自然跳过
    """
    document: Document
    reconciliation: ReconciliationPayload
    grounding: GroundingTargets | None = None


class GroundingDTOIR(DTOIRBase):
    """IR2：供 GroundingEngine 消费。不携带 document 元数据。"""
    grounding: GroundingTargets